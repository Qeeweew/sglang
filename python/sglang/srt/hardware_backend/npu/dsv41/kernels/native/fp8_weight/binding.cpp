#include "torch.h"
#include "launch.h"
#include "format.h"
#include "jit.h"

namespace native_host {
namespace {
using namespace fp8_weight;
void fp8_weight(const at::Tensor &codes, const at::Tensor &scales, const at::Tensor &out,
                const at::Tensor &lut, const at::Tensor &groups) {
    check(codes, at::kByte, codes.device());
    TORCH_CHECK_VALUE(codes.dim() == 2, "FP8 weight requires a matrix");
    check(scales, at::kByte, codes.device(),
          {(codes.size(0) + kScaleGroup - 1) / kScaleGroup,
           (codes.size(1) + kScaleGroup - 1) / kScaleGroup});
    check(out, at::kBFloat16, codes.device(), codes.sizes());
    check(lut, at::kFloat, codes.device(), {kLutElements});
    check(groups, at::kInt, codes.device(), {kTileColumns});
    disjoint({out}, {codes, scales, lut, groups});
    if (!out.numel())
        return;

    const int64_t rows = codes.size(0);
    const int64_t cols = codes.size(1);
    const int64_t blocks = core_count(codes.get_device());

    c10_npu::NPUGuard guard(codes.device());
    auto stream = c10_npu::getCurrentNPUStream(codes.get_device());
    track(codes, stream);
    track(scales, stream);
    track(out, stream);
    track(lut, stream);
    track(groups, stream);
    dsv41_fp8_weight_launch(stream.stream(), u32(blocks), ptr(codes), ptr(scales), ptr(out),
                            ptr(lut), ptr(groups), rows, cols);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("prepare_fp8_weight(Tensor codes, Tensor scales, Tensor out, Tensor lut, Tensor groups) "
          "-> ()",
          [](const at::Tensor &codes, const at::Tensor &scales, const at::Tensor &out,
             const at::Tensor &lut, const at::Tensor &groups) {
              native_jit::PrepareOnlyScope prepare;
              fp8_weight(codes, scales, out, lut, groups);
          });
    m.def(
        "fp8_weight(Tensor codes, Tensor scales, Tensor(a!) out, Tensor lut, Tensor groups) -> ()",
        &fp8_weight);
}
} // namespace native_host
