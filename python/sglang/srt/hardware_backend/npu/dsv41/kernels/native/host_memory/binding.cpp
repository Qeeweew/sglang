// Registered host allocations are Tensor-owned, including views held by loaders.
#include "torch.h"
#include <numaif.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <memory>

namespace native_host {
namespace {
struct Mapping {
    void *pointer = nullptr;
    size_t bytes;
    int device;
    int fd = -1;
    bool registered = false, acl_allocation;
    Mapping(size_t n, int dev, bool use_acl) : bytes(n), device(dev), acl_allocation(use_acl) {}
    ~Mapping() {
        if (!pointer) {
            if (fd >= 0)
                ::close(fd);
            return;
        }
        try {
            c10_npu::NPUGuard guard(c10::Device(c10::DeviceType::PrivateUse1, device));
            if (registered) {
                acl_check(aclrtSynchronizeDevice(), "retire host DMA");
                acl_check(aclrtHostUnregister(pointer), "aclrtHostUnregister");
            }
            if (acl_allocation)
                acl_check(aclrtFreeHost(pointer), "aclrtFreeHost");
            else
                TORCH_CHECK(munmap(pointer, bytes) == 0, "munmap: ", strerror(errno));
            if (fd >= 0)
                ::close(fd);
        } catch (const std::exception &e) {
            // Never unmap still-registered memory after a failed retirement.
            TORCH_WARN("host mapping retirement failed: ", e.what());
        }
    }
};
class HostBuffer : public torch::CustomClassHolder {
    at::Tensor tensor_;
    void initialize(std::vector<int64_t> shape, int64_t device, int64_t node, bool use_acl,
                    int64_t shared_fd) {
        TORCH_CHECK(!shape.empty() && device >= 0 && device <= INT16_MAX && node >= -1,
                    "invalid registered host allocation");
        int64_t bytes = 1;
        for (auto extent : shape) {
            TORCH_CHECK(extent > 0 && extent <= INT64_MAX / bytes, "invalid host byte extent");
            bytes *= extent;
        }
        c10_npu::NPUGuard guard(c10::Device(c10::DeviceType::PrivateUse1, device));
        auto mapping = std::make_shared<Mapping>(bytes, device, use_acl);
        if (shared_fd >= 0) {
            TORCH_CHECK(
                !use_acl && node == -1 && shared_fd <= INT_MAX,
                "shared host memory requires a valid file descriptor and no private NUMA policy");
            mapping->fd = fcntl(shared_fd, F_DUPFD_CLOEXEC, 0);
            TORCH_CHECK(mapping->fd >= 0, "duplicate shared host fd: ", strerror(errno));
            struct stat status;
            TORCH_CHECK(fstat(mapping->fd, &status) == 0 && S_ISREG(status.st_mode) &&
                            status.st_size >= bytes,
                        "shared host file does not cover the requested tensor");
            mapping->pointer =
                mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
            if (mapping->pointer == MAP_FAILED) {
                mapping->pointer = nullptr;
                TORCH_CHECK(false, "shared mmap: ", strerror(errno));
            }
            madvise(mapping->pointer, bytes, MADV_HUGEPAGE);
        } else if (use_acl) {
            TORCH_CHECK(node == -1, "NUMA policy requires mmap allocation");
            acl_check(aclrtMallocHost(&mapping->pointer, bytes), "aclrtMallocHost");
        } else {
            mapping->pointer =
                mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mapping->pointer == MAP_FAILED) {
                mapping->pointer = nullptr;
                TORCH_CHECK(false, "mmap: ", strerror(errno));
            }
        }
        if (node >= 0) {
            constexpr int bits = sizeof(unsigned long) * 8;
            std::vector<unsigned long> mask(node / bits + 1);
            mask[node / bits] = 1UL << (node % bits);
            TORCH_CHECK(
                mbind(mapping->pointer, bytes, MPOL_BIND, mask.data(), mask.size() * bits, 0) == 0,
                "mbind: ", strerror(errno));
            // Fault under the VMA policy before registration, without changing worker affinity.
            memset(mapping->pointer, 0, bytes);
        }
        acl_check(
            aclrtHostRegisterV2(mapping->pointer, bytes, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED),
            "aclrtHostRegisterV2");
        mapping->registered = true;
        tensor_ = at::from_blob(
            mapping->pointer, shape, [mapping](void *) {},
            at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    }

  public:
    HostBuffer() = default;
    HostBuffer(std::vector<int64_t> shape, int64_t device, int64_t node, bool use_acl) {
        initialize(std::move(shape), device, node, use_acl, -1);
    }
    static c10::intrusive_ptr<HostBuffer> from_shared_fd(std::vector<int64_t> shape, int64_t device,
                                                         int64_t fd) {
        TORCH_CHECK(fd >= 0, "shared host file descriptor must be nonnegative");
        auto owner = c10::make_intrusive<HostBuffer>();
        owner->initialize(std::move(shape), device, -1, false, fd);
        return owner;
    }
    at::Tensor tensor() const {
        TORCH_CHECK(tensor_.defined(), "host buffer is closed");
        return tensor_;
    }
    void close() {
        tensor_ = at::Tensor();
    }
};
std::vector<int64_t> page_nodes(const at::Tensor &tensor, int64_t count) {
    TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous() && tensor.numel() && count > 0,
                "page query requires a live contiguous CPU tensor");
    int64_t bytes = tensor.numel() * tensor.element_size(), page = sysconf(_SC_PAGESIZE);
    int64_t pages = (bytes + page - 1) / page;
    count = std::min(count, pages);
    std::vector<void *> addresses;
    std::vector<int> status(count);
    for (int64_t i = 0; i < count; ++i)
        addresses.push_back(static_cast<char *>(tensor.data_ptr()) + (pages * i / count) * page);
    TORCH_CHECK(move_pages(0, count, addresses.data(), nullptr, status.data(), 0) == 0,
                "move_pages: ", strerror(errno));
    return std::vector<int64_t>(status.begin(), status.end());
}
int64_t physical_device(int64_t device) {
    int32_t physical = -1;
    acl_check(aclrtGetPhyDevIdByUserDevId(device, &physical), "aclrtGetPhyDevIdByUserDevId");
    return physical;
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.class_<HostBuffer>("HostBuffer")
        .def(torch::init<std::vector<int64_t>, int64_t, int64_t, bool>())
        .def_static("from_shared_fd", &HostBuffer::from_shared_fd)
        .def("tensor", &HostBuffer::tensor)
        .def("close", &HostBuffer::close);
    m.def("page_nodes(Tensor tensor, int count) -> int[]", &page_nodes);
    m.def("physical_device(int device) -> int", &physical_device);
}
} // namespace native_host
