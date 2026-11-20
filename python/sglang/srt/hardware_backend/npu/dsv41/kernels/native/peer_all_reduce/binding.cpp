#include "torch.h"
#include "launch.h"
#include "layout.h"
#include <dlfcn.h>

namespace native_host {
namespace {
using OptionalTensor = std::optional<at::Tensor>;
using CloseImport = int (*)(void *);
class PeerCommunicator : public torch::CustomClassHolder {
    int device_, rank_, world_, attention_tp_;
    void *local_ = nullptr;
    std::vector<void *> peers_;
    at::Tensor table_;
    std::string key_;
    CloseImport close_import_;
    c10::Device device() const {
        return c10::Device(c10::DeviceType::PrivateUse1, device_);
    }
    void connected() const {
        TORCH_CHECK(local_ && table_.defined(), "peer communicator is not connected");
    }
    void rows(int64_t count) const {
        TORCH_CHECK(count > 0 && count <= (world_ == 2 ? 128 : 64 * attention_tp_),
                    "peer communication exceeds its local row capacity");
    }
    void matrix(const at::Tensor &x, int64_t n, at::ScalarType dtype = at::kBFloat16) const {
        check(x, dtype, device(), {n, 5120});
    }
    void shared_input(const OptionalTensor &shared, int64_t n) const {
        if (shared)
            matrix(*shared, attention_tp_ == 1 ? n : (rank_ % 2 ? n / 2 : (n + 1) / 2));
    }
    int64_t routes(const at::Tensor &values, const at::Tensor &inverse, const at::Tensor &ids,
                   const at::Tensor &weights, int64_t n) const {
        check(ids, at::kInt, device());
        TORCH_CHECK(ids.dim() == 2 && ids.size(0) == (8 / attention_tp_) * n && ids.size(1) > 0 &&
                        ids.size(1) <= 16,
                    "expected global token route IDs [DP*rows,topk<=16]");
        check(inverse, at::kInt, device(), {ids.numel()});
        check(weights, at::kFloat, device(), ids.sizes());
        check(values, values.scalar_type() == at::kBFloat16 ? at::kBFloat16 : at::kHalf, device());
        TORCH_CHECK(values.dim() == 2 && values.size(1) == 5120 && values.size(0) <= ids.numel(),
                    "invalid routed values");
        return ids.size(1);
    }
    c10_npu::NPUStream stream() const {
        auto stream = c10_npu::getCurrentNPUStream(device_);
        record(table_, stream);
        return stream;
    }

  public:
    PeerCommunicator(int64_t device, int64_t rank, int64_t world, int64_t attention_tp)
        : device_(device), rank_(rank), world_(world), attention_tp_(attention_tp),
          peers_(world, nullptr) {
        TORCH_CHECK((world == 2 || world == 8) && rank >= 0 && rank < world,
                    "peer communicator requires a TP2 or EP8 group");
        TORCH_CHECK(attention_tp == 1 || attention_tp == 2, "attention TP must be 1 or 2");
        TORCH_CHECK(world != 2 || attention_tp == 2, "SUM requires attention TP2");
        c10_npu::NPUGuard guard(this->device());
        close_import_ = reinterpret_cast<CloseImport>(dlsym(RTLD_DEFAULT, "rtIpcCloseMemory"));
        TORCH_CHECK(close_import_, "CANN rtIpcCloseMemory is required");
        if (world == 2)
            prepare_peer_all_reduce();
        else
            prepare_peer_moe(attention_tp_);
        prepare_peer_mhc(world, attention_tp_);
        const auto bytes =
            world == 2 ? peer_all_reduce::kAllocationBytes : peer_all_reduce::kMoeAllocationBytes;
        acl_check(aclrtMalloc(&local_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate peer region");
        try {
            acl_check(aclrtMemset(local_, bytes, 0, peer_all_reduce::kControlBytes),
                      "initialize peer control");
            char key[65]{};
            acl_check(aclrtIpcMemGetExportKey(local_, bytes, key, sizeof(key), 0),
                      "export peer region");
            key_ = key;
        } catch (...) {
            aclrtFree(local_);
            local_ = nullptr;
            throw;
        }
    }
    ~PeerCommunicator() override {
        try {
            close();
        } catch (const std::exception &e) {
            TORCH_WARN("peer communication retirement failed: ", e.what());
        }
    }
    std::string key() const {
        return key_;
    }
    int64_t pid() const {
        c10_npu::NPUGuard guard(device());
        int32_t pid;
        acl_check(aclrtDeviceGetBareTgid(&pid), "peer bare TGID");
        return pid;
    }
    void allow(const std::vector<int64_t> &pids) {
        c10_npu::NPUGuard guard(device());
        std::vector<int32_t> bare_pids;
        for (auto pid : pids) {
            TORCH_CHECK(pid > 0 && pid <= INT32_MAX, "invalid peer TGID");
            bare_pids.push_back(pid);
        }
        acl_check(aclrtIpcMemSetImportPid(key_.c_str(), bare_pids.data(), bare_pids.size()),
                  "allow peer TGIDs");
    }
    void connect(const std::vector<std::string> &keys) {
        TORCH_CHECK(local_ && !table_.defined() && keys.size() == size_t(world_),
                    "invalid peer connection");
        c10_npu::NPUGuard guard(device());
        for (int rank = 0; rank < world_; ++rank) {
            if (rank == rank_)
                peers_[rank] = local_;
            else
                acl_check(aclrtIpcMemImportByKey(&peers_[rank], keys[rank].c_str(), 1),
                          "import peer region");
        }
        table_ = at::empty({world_}, at::TensorOptions().device(device()).dtype(at::kLong));
        std::vector<uint64_t> pointers;
        for (auto ptr : peers_)
            pointers.push_back(reinterpret_cast<uint64_t>(ptr));
        acl_check(aclrtMemcpy(table_.data_ptr(), world_ * 8, pointers.data(), world_ * 8,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "initialize immutable peer address table");
    }
    void reduce(const at::Tensor &input, const at::Tensor &output) {
        connected();
        TORCH_CHECK(world_ == 2 && input.dim() == 2, "expected TP2 matrix");
        rows(input.size(0));
        matrix(input, input.size(0));
        matrix(output, input.size(0));
        disjoint({output}, {input});
        c10_npu::NPUGuard guard(device());
        auto current = stream();
        record(input, current);
        record(output, current);
        launch_peer_all_reduce(current.stream(), input.data_ptr(), output.data_ptr(), local_,
                               peers_[1 - rank_], table_.data_ptr(), input.numel());
    }
    void gather(const at::Tensor &input, const at::Tensor &output) {
        connected();
        TORCH_CHECK(world_ == 8 && input.dim() == 2, "expected EP8 matrix");
        rows(input.size(0));
        matrix(input, input.size(0));
        matrix(output, (8 / attention_tp_) * input.size(0));
        disjoint({output}, {input});
        c10_npu::NPUGuard guard(device());
        auto current = stream();
        record(input, current);
        record(output, current);
        launch_peer_moe(current.stream(), input.data_ptr(), nullptr, nullptr, nullptr, nullptr,
                        output.data_ptr(), local_, table_.data_ptr(), rank_, input.size(0), 0, true,
                        false, attention_tp_);
    }
    void return_sum(const at::Tensor &partial, const OptionalTensor &shared,
                    const at::Tensor &output) {
        connected();
        TORCH_CHECK(world_ == 8 && output.dim() == 2, "expected EP8 output matrix");
        const int64_t n = output.size(0);
        rows(n);
        matrix(partial, (8 / attention_tp_) * n);
        matrix(output, n);
        shared_input(shared, n);
        disjoint({output}, {partial, shared});
        c10_npu::NPUGuard guard(device());
        auto current = stream();
        record(partial, current);
        track(shared, current);
        record(output, current);
        launch_peer_moe(current.stream(), partial.data_ptr(), nullptr, nullptr, nullptr,
                        ptr(shared), output.data_ptr(), local_, table_.data_ptr(), rank_, n, 0,
                        false, false, attention_tp_);
    }
    void finalize_return(const at::Tensor &values, const at::Tensor &inverse, const at::Tensor &ids,
                         const at::Tensor &weights, const OptionalTensor &shared,
                         const at::Tensor &output) {
        connected();
        TORCH_CHECK(world_ == 8 && output.dim() == 2, "expected EP8 output matrix");
        const int64_t n = output.size(0);
        rows(n);
        matrix(output, n);
        shared_input(shared, n);
        const int64_t topk = routes(values, inverse, ids, weights, n);
        disjoint({output}, {values, inverse, ids, weights, shared});
        c10_npu::NPUGuard guard(device());
        auto current = stream();
        for (auto &x : {values, inverse, ids, weights, output})
            record(x, current);
        track(shared, current);
        launch_peer_moe(current.stream(), values.data_ptr(), inverse.data_ptr(), ids.data_ptr(),
                        weights.data_ptr(), ptr(shared), output.data_ptr(), local_,
                        table_.data_ptr(), rank_, n, topk, false, true, attention_tp_, values.scalar_type() == at::kBFloat16);
    }
    void post(const at::Tensor &input, const at::Tensor &residual, const at::Tensor &pre,
              const at::Tensor &post_mix, const at::Tensor &comb, const OptionalTensor &weight,
              const at::Tensor &updated, const OptionalTensor &normalized,
              const OptionalTensor &inverse, const OptionalTensor &ids,
              const OptionalTensor &weights, const OptionalTensor &shared) {
        connected();
        TORCH_CHECK(residual.dim() == 3 && residual.size(1) == 4 && residual.size(2) == 5120,
                    "expected BF16 residual [rows,4,5120]");
        const int64_t n = residual.size(0);
        rows(n);
        check(residual, at::kBFloat16, device());
        check(updated, at::kBFloat16, device(), residual.sizes());
        check(pre, at::kFloat, device(), {n, 4});
        check(post_mix, at::kFloat, device(), {n, 4});
        check(comb, at::kFloat, device(), {n, 4, 4});
        TORCH_CHECK(bool(weight) == bool(normalized),
                    "normalization requires weight and output together");
        if (weight) {
            check(*weight, at::kFloat, device(), {5120});
            matrix(*normalized, n);
        }
        int64_t topk = 0;
        if (world_ == 2) {
            matrix(input, n);
            TORCH_CHECK(weight, "attention boundary requires normalization");
        } else {
            TORCH_CHECK(inverse && ids && weights, "MoE post requires deferred route data");
            topk = routes(input, *inverse, *ids, *weights, n);
            shared_input(shared, n);
        }
        disjoint({updated, normalized},
                 {input, residual, pre, post_mix, comb, weight, inverse, ids, weights, shared});
        c10_npu::NPUGuard guard(device());
        auto current = stream();
        for (auto &x : {input, residual, pre, post_mix, comb, updated})
            record(x, current);
        for (auto &x : {weight, normalized, inverse, ids, weights, shared})
            track(x, current);
        launch_peer_mhc(current.stream(), input.data_ptr(), residual.data_ptr(), pre.data_ptr(),
                        post_mix.data_ptr(), comb.data_ptr(), ptr(weight), updated.data_ptr(),
                        ptr(normalized), local_, table_.data_ptr(), ptr(inverse), ptr(ids),
                        ptr(weights), ptr(shared), rank_, n, topk, world_, bool(weight),
                        attention_tp_, world_ == 8 && input.scalar_type() == at::kBFloat16);
    }
    void disconnect() {
        if (!local_)
            return;
        c10_npu::NPUGuard guard(device());
        acl_check(aclrtSynchronizeDevice(), "retire peer communication");
        table_ = at::Tensor();
        for (int rank = 0; rank < world_; ++rank)
            if (rank != rank_ && peers_[rank]) {
                acl_check(close_import_(peers_[rank]), "close imported peer region");
                peers_[rank] = nullptr;
            }
    }
    void close() {
        if (!local_)
            return;
        disconnect();
        c10_npu::NPUGuard guard(device());
        acl_check(aclrtIpcMemClose(key_.c_str()), "close peer export");
        acl_check(aclrtFree(local_), "free peer region");
        local_ = nullptr;
        key_.clear();
    }
};
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.class_<PeerCommunicator>("PeerCommunicator")
        .def(torch::init<int64_t, int64_t, int64_t, int64_t>())
        .def("key", &PeerCommunicator::key)
        .def("pid", &PeerCommunicator::pid)
        .def("allow", &PeerCommunicator::allow)
        .def("connect", &PeerCommunicator::connect)
        .def("reduce", &PeerCommunicator::reduce)
        .def("gather", &PeerCommunicator::gather)
        .def("return_sum", &PeerCommunicator::return_sum)
        .def("finalize_return", &PeerCommunicator::finalize_return)
        .def("post", &PeerCommunicator::post)
        .def("disconnect", &PeerCommunicator::disconnect)
        .def("close", &PeerCommunicator::close);
}
} // namespace native_host
