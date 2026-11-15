#include "torch.h"
#include "jit.h"
#include "launch.h"

namespace native_host {
namespace {
void quantize_weight(const at::Tensor &weight, const at::Tensor &out, const at::Tensor &scales) {
    TORCH_CHECK_VALUE(weight.dim() == 2, "weight quantization requires [N,K]");
    auto rows = weight.size(0), k = weight.size(1);
    TORCH_CHECK_VALUE(rows > 0 && k > 0 && k <= INT32_MAX && k % w8a16::kGroup == 0,
                      "weight quantization requires positive N and K256 alignment");
    auto device = weight.device();
    check(weight, at::kBFloat16, device);
    check(out, at::kChar, device, {rows, k});
    check(scales, at::kHalf, device, {k / w8a16::kGroup, rows});
    disjoint({out, scales}, {weight});
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    for (const auto &tensor : {weight, out, scales}) record(tensor, stream);
    w8a16::QuantizeSources sources{uintptr_t(ptr(weight)), uintptr_t(ptr(out)),
                                   uintptr_t(ptr(scales)), rows};
    w8a16_quantize_launch(stream.stream(), sources, int(k), core_count(device.index()));
}

void gemm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &s,
          const at::Tensor &out, const at::Tensor &workspace, int64_t bm, int64_t bn, int64_t bk) {
    TORCH_CHECK_VALUE(x.dim() == 2 || x.dim() == 3, "W8A16 input requires [M,K] or [M,B,K]");
    const auto m = x.size(0), k = x.size(-1), batches = x.dim() == 3 ? x.size(1) : 1;
    TORCH_CHECK_VALUE(w.dim() == 3 && w.size(2) == w8a16::kC0 && batches > 0 &&
                          w.size(0) % batches == 0,
                      "W8A16 repacked weights require [B*N/16,K,16]");
    const auto n = w.size(0) / batches * w8a16::kC0;
    TORCH_CHECK_VALUE(k > 0 && k <= UINT16_MAX && k % w8a16::kGroup == 0 &&
                          n > 0 && n <= INT32_MAX && n % 128 == 0 &&
                          batches * k <= UINT16_MAX && uint64_t(batches) * n <= UINT32_MAX / 2,
                      "W8A16 requires G256 K, N128, uint16 ND2NZ input stride and uint32 output byte stride");
    TORCH_CHECK_VALUE((bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256) &&
                          (bn == 64 || bn == 128 || bn == 256) && n % bn == 0 &&
                          (bk == 128 || bk == 256 || bk == 512) && k % bk == 0,
                      "invalid W8A16 tile plan");
    auto device = x.device();
    TORCH_CHECK_VALUE(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                      "W8A16 requires BF16/FP16 activation and output");
    check(x, x.scalar_type(), device);
    check(w, at::kChar, device, {batches * n / w8a16::kC0, k, w8a16::kC0});
    check(s, at::kHalf, device, {batches * n / w8a16::kC0, k / w8a16::kGroup, w8a16::kC0});
    check(out, x.scalar_type(), device);
    TORCH_CHECK_VALUE(out.sizes() == (x.dim() == 3 ? at::IntArrayRef({m, batches, n})
                                                : at::IntArrayRef({m, n})),
                      "W8A16 output shape differs");
    check(workspace, x.scalar_type(), device);
    for (const auto &tensor : {x, w, s, out, workspace})
        TORCH_CHECK_VALUE(uintptr_t(ptr(tensor)) % (w8a16::kC0 * sizeof(at::Half)) == 0,
                          "W8A16 GM pointers require 32-byte alignment");
    const bool bf16 = x.scalar_type() == at::kBFloat16;
    const w8a16::Storage storage(bm, bn, bk, bf16);
    TORCH_CHECK_VALUE(storage.fits(), "W8A16 tile exceeds arch2201 resource budget");
    c10_npu::NPUGuard guard(device);
    const auto workers = core_count(device.index(), true);
    TORCH_CHECK_VALUE(core_count(device.index()) == w8a16::kVectorActors * workers,
                      "W8A16 requires two AIV per AIC");
    TORCH_CHECK_VALUE(workspace.numel() >= workers * w8a16::kSlots * bn * bk,
                      "W8A16 workspace is smaller than the persistent plan");
    disjoint({out, workspace}, {x, w, s});
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    for (const auto &t : {x, w, s, out, workspace}) record(t, stream);
    w8a16::Arguments a{uintptr_t(ptr(x)), uintptr_t(ptr(w)), uintptr_t(ptr(s)),
                       uintptr_t(ptr(out)), uintptr_t(ptr(workspace)), m,
                       int(k), int(n), int(batches), int(bm), int(bn), int(bk), int(workers), bf16};
    w8a16_launch(stream.stream(), a);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("quantize_g256_weight(Tensor weight, Tensor(a!) out, Tensor(b!) scales) -> ()",
          &quantize_weight);
    m.def("w8a16(Tensor x, Tensor weight, Tensor scales, Tensor(a!) out, Tensor(b!) workspace, "
          "int bm, int bn, int bk) -> ()", &gemm);
    m.def("prepare_w8a16(Tensor x, Tensor weight, Tensor scales, Tensor(a!) out, Tensor(b!) workspace, "
          "int bm, int bn, int bk) -> ()",
          [](const at::Tensor &x, const at::Tensor &w, const at::Tensor &s,
             const at::Tensor &out, const at::Tensor &ws, int64_t bm, int64_t bn, int64_t bk) {
              native_jit::PrepareOnlyScope prepare;
              gemm(x, w, s, out, ws, bm, bn, bk);
          });
}
} // namespace native_host
