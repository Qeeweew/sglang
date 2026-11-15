#include "torch.h"
#include "launch.h"
#include "format.h"
#include "jit.h"

namespace native_host {
namespace {
void engram_gather(const at::Tensor &codes, const at::Tensor &scales, const at::Tensor &ids,
                   const at::Tensor &out, const at::Tensor &tables) {
    check(ids, at::kLong, ids.device());
    TORCH_CHECK_VALUE(
        codes.dim() == 2 && codes.size(1) > 0 && codes.size(1) % engram::kScaleGroupElements == 0 &&
            codes.scalar_type() == at::kByte && codes.is_contiguous() &&
            scales.sizes() ==
                at::IntArrayRef({codes.size(0), codes.size(1) / engram::kScaleGroupElements}) &&
            scales.scalar_type() == at::kByte && scales.is_contiguous(),
        "expected byte codes[vocab,width], scales[vocab,width/32]");
    for (const auto &t : {codes, scales})
        TORCH_CHECK_VALUE(t.device().is_cpu() || t.device() == ids.device(),
                          "Engram storage belongs to another NPU");
    auto shape = ids.sizes().vec();
    shape.push_back(codes.size(1));
    check(out, at::kBFloat16, ids.device(), shape);
    check(tables, at::kFloat, ids.device(), {engram::kCodebookElements});
    disjoint({out}, {ids, codes, scales, tables});
    if (!ids.numel())
        return;

    const int64_t rows = ids.numel();
    const int64_t vocab = codes.size(0);
    const int64_t blocks = core_count(ids.get_device());

    c10_npu::NPUGuard guard(ids.device());
    auto stream = c10_npu::getCurrentNPUStream(ids.get_device());
    track(ids, stream);
    track(out, stream);
    track(tables, stream);
    if (!codes.device().is_cpu())
        track(codes, stream);
    if (!scales.device().is_cpu())
        track(scales, stream);
    dsv41_engram_launch(stream.stream(), u32(blocks), mapped(codes), mapped(scales), ptr(ids),
                        ptr(out), ptr(tables), rows, vocab, u32(codes.size(1)));
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("engram(Tensor codes, Tensor scales, Tensor ids, Tensor(a!) out, Tensor tables) -> ()",
          &engram_gather);
    m.def(
        "prepare_engram(Tensor codes, Tensor scales, Tensor ids, Tensor out, Tensor tables) -> ()",
        [](const at::Tensor &codes, const at::Tensor &scales, const at::Tensor &ids,
           const at::Tensor &out, const at::Tensor &tables) {
            native_jit::PrepareOnlyScope prepare;
            engram_gather(codes, scales, ids, out, tables);
        });
}
} // namespace native_host
