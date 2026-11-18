#include "launch.h"
#include "torch.h"

namespace native_host {
namespace {
at::Tensor local_greedy(const at::Tensor &base, const at::Tensor &bias, int64_t start) {
    const auto device = base.device();
    TORCH_CHECK_VALUE(device.type() == c10::DeviceType::PrivateUse1 && bias.device() == device,
                      "base and bias must share an NPU");
    for (const auto &tensor : {base, bias}) {
        const auto format = at_npu::native::get_npu_format(tensor);
        TORCH_CHECK_VALUE(tensor.dim() == 2 && tensor.stride(1) == 1 &&
                          (tensor.scalar_type() == at::kBFloat16 || tensor.scalar_type() == at::kFloat) &&
                          (format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW),
                          "expected row-strided ND BF16/FP32 matrices");
    }
    const auto rows = base.size(0), width = bias.size(1);
    TORCH_CHECK_VALUE(rows == bias.size(0) && width <= base.size(1) && start >= 0 &&
                      start <= INT32_MAX - width, "invalid vocabulary shard");
    c10_npu::NPUGuard guard(device);
    auto out = at::empty({rows, 2}, base.options().dtype(at::kFloat));
    if (!rows) return out;
    const auto parts = std::max<int64_t>(1, (width + dspark_markov::kColumns - 1) / dspark_markov::kColumns);
    auto partial = parts == 1 ? out : at::empty({parts, rows, 2}, out.options());
    auto stream = c10_npu::getCurrentNPUStream(base.get_device());
    for (const auto &tensor : {base, bias, partial, out}) record(tensor, stream);
    const uint32_t formats = (base.scalar_type() == at::kFloat) | ((bias.scalar_type() == at::kFloat) << 1);
    dspark_markov::select(stream.stream(), ptr(base), ptr(bias), ptr(partial), rows, width, parts,
                          base.stride(0), bias.stride(0), uint32_t(start), formats);
    if (parts > 1) dspark_markov::launch(stream.stream(), ptr(partial), ptr(out), parts, rows, true);
    return out;
}

at::Tensor merge_candidates(const at::Tensor &candidates) {
    const auto device = candidates.device();
    check(candidates, at::kFloat, device);
    TORCH_CHECK_VALUE(candidates.dim() == 3 && candidates.size(0) > 0 && candidates.size(2) == 2,
                      "expected [ranks, rows, 2] FP32 (score, bitcast token ID)");
    c10_npu::NPUGuard guard(device);
    const auto rows = candidates.size(1);
    auto tokens = at::empty({rows}, candidates.options().dtype(at::kLong));
    if (rows) {
        auto stream = c10_npu::getCurrentNPUStream(candidates.get_device());
        record(candidates, stream);
        record(tokens, stream);
        dspark_markov::launch(stream.stream(), ptr(candidates), ptr(tokens), candidates.size(0), rows);
    }
    return tokens;
}
}

TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("markov_local_greedy(Tensor base, Tensor bias, int vocab_start) -> Tensor", &local_greedy);
    m.def("markov_merge_candidates(Tensor candidates) -> Tensor", &merge_candidates);
}
} // namespace native_host
