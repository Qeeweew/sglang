#pragma once
#include <ATen/ATen.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

namespace native_host {
inline void acl_check(aclError code, const char *operation) {
    TORCH_CHECK(code == ACL_SUCCESS, operation, " failed: ACL ", code);
}
inline void record(const at::Tensor &tensor, const c10_npu::NPUStream &stream) {
    if (tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.numel()) {
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
        // The allocator erases only on the allocation stream. Keep every
        // foreign-stream use, including HCCL's independent records.
        if (stream == c10_npu::getCurrentNPUStream(tensor.device().index()))
            c10_npu::NPUCachingAllocator::eraseStream(tensor.storage().data_ptr(), stream);
    }
}
// Cached per host thread/device; no driver queries in a warmed launch.
inline int64_t core_count(int device, bool cube = false) {
    thread_local std::unordered_map<int, std::array<int64_t, 2>> cache;
    auto found = cache.find(device);
    if (found == cache.end()) {
        std::array<int64_t, 2> counts{};
        acl_check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &counts[0]),
                  "vector cores");
        acl_check(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &counts[1]), "cube cores");
        TORCH_CHECK_VALUE(counts[0] > 0 && counts[1] > 0, "invalid NPU core counts");
        found = cache.emplace(device, counts).first;
    }
    return found->second[cube ? 1 : 0];
}
inline void check(const at::Tensor &t, at::ScalarType type, const at::Device &device) {
    TORCH_CHECK_VALUE(t.device() == device && device.type() == c10::DeviceType::PrivateUse1 &&
                          t.scalar_type() == type && t.is_contiguous(),
                      "expected contiguous ", type, " on ", device);
    const auto format = at_npu::native::get_npu_format(t);
    TORCH_CHECK_VALUE(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                      "expected ND storage (format 0 or 2)");
}
inline void check(const at::Tensor &t, at::ScalarType type, const at::Device &device,
                  at::IntArrayRef shape) {
    check(t, type, device);
    TORCH_CHECK_VALUE(t.sizes() == shape, "unexpected tensor shape: ", t.sizes(), "; expected ",
                      shape);
}
inline bool overlaps(const at::Tensor &a, const at::Tensor &b) {
    if (!a.numel() || !b.numel() || a.device() != b.device())
        return false;
    auto extent = [](const at::Tensor &t) {
        int64_t count = 1;
        for (int64_t i = 0; i < t.dim(); ++i)
            count += (t.size(i) - 1) * t.stride(i);
        return count * t.element_size();
    };
    const auto ab = reinterpret_cast<uintptr_t>(a.data_ptr()),
               bb = reinterpret_cast<uintptr_t>(b.data_ptr());
    return ab < bb + extent(b) && bb < ab + extent(a);
}
inline void disjoint(std::initializer_list<std::optional<at::Tensor>> outputs,
                     std::initializer_list<std::optional<at::Tensor>> inputs = {}) {
    std::vector<at::Tensor> seen;
    for (const auto &t : inputs)
        if (t)
            seen.push_back(*t);
    for (const auto &t : outputs)
        if (t) {
            for (const auto &s : seen)
                TORCH_CHECK_VALUE(!overlaps(*t, s), "output/workspace aliases input/output");
            seen.push_back(*t);
        }
}
inline uint8_t *ptr(const at::Tensor &tensor) {
    return static_cast<uint8_t *>(tensor.data_ptr());
}
inline uint8_t *ptr(const std::optional<at::Tensor> &tensor) {
    return tensor ? ptr(*tensor) : nullptr;
}
inline void track(const at::Tensor &tensor, const c10_npu::NPUStream &stream) {
    TORCH_CHECK_VALUE(tensor.device() == stream.device(), "native tensors must share one NPU");
    record(tensor, stream);
}
inline void track(const std::optional<at::Tensor> &tensor, const c10_npu::NPUStream &stream) {
    if (tensor)
        track(*tensor, stream);
}
inline uint32_t u32(int64_t value) {
    TORCH_CHECK_VALUE(value >= 0 && value <= UINT32_MAX, "native scalar exceeds UINT32");
    return static_cast<uint32_t>(value);
}
inline float f32(double value) {
    // Preserve each operator's IEEE scalar contract (DSpark epsilon includes
    // nonfinite boundary tests); semantic validation belongs to its interface.
    return static_cast<float>(value);
}

inline uint8_t *mapped(const at::Tensor &tensor) {
    if (!tensor.device().is_cpu())
        return ptr(tensor);
    void *address = nullptr;
    acl_check(aclrtHostGetDevicePointer(tensor.data_ptr(), &address, 0),
              "aclrtHostGetDevicePointer");
    return static_cast<uint8_t *>(address);
}
} // namespace native_host
