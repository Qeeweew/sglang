#include "torch.h"
#include "launch.h"
#include "layout.h"
#include "jit.h"

namespace native_host {
namespace {
void engram_hash(const at::Tensor &ids, const at::Tensor &pos, const at::Tensor &cu,
                 const at::Tensor &slots, const at::Tensor &history, const at::Tensor &active,
                 const at::Tensor &map, const at::Tensor &mult, const at::Tensor &primes,
                 const at::Tensor &offsets, const at::Tensor &out, int64_t pad, int64_t image,
                 int64_t mm_shift) {
    const auto device = ids.device();
    check(ids, at::kInt, device);
    check(slots, at::kInt, device);
    check(history, at::kInt, device);
    TORCH_CHECK_VALUE(ids.dim() == 1 && slots.dim() == 1 && history.dim() == 2 &&
                          history.size(1) > 0,
                      "invalid Engram hash row axes");
    check(pos, at::kLong, device, {ids.numel()});
    check(cu, at::kInt, device, {slots.numel() + 1});
    check(active, at::kInt, device, {ids.numel()});
    check(map, at::kInt, device);
    TORCH_CHECK_VALUE(map.dim() == 1 && map.numel(), "invalid token map");
    check(mult, at::kLong, device);
    check(primes, at::kLong, device);
    TORCH_CHECK_VALUE(mult.dim() == 2 && mult.size(0) > 0 && mult.size(1) == history.size(1) + 1 &&
                          primes.dim() == 3 && primes.size(0) == mult.size(0) &&
                          primes.size(1) == history.size(1) && primes.size(2) > 0,
                      "invalid n-gram multiplier/prime axes");
    const int64_t layers = mult.size(0), heads = primes.size(2);
    const int64_t hash_columns = history.size(1) * heads;
    TORCH_CHECK_VALUE(hash_columns <= int64_t(arch2201::kUbBytes / sizeof(int64_t)),
                      "n-gram output row exceeds UB budget");
    TORCH_CHECK_VALUE(engram_ops::HashLayout(u32(hash_columns)).bytes <= arch2201::kUbBytes,
                      "n-gram result row exceeds UB budget");
    check(offsets, at::kLong, device, {layers, hash_columns});
    check(out, at::kLong, device, {layers, ids.numel(), hash_columns});
    disjoint({out}, {ids, pos, cu, slots, history, active, map, mult, primes, offsets});
    if (!ids.numel())
        return;

    const int64_t rows = ids.numel();
    const int64_t batch = slots.numel();
    const int64_t history_rows = history.size(0);
    const int64_t vocab = map.numel();
    const int64_t blocks = core_count(ids.get_device());

    c10_npu::NPUGuard guard(ids.device());
    auto stream = c10_npu::getCurrentNPUStream(ids.get_device());
    track(ids, stream);
    track(pos, stream);
    track(cu, stream);
    track(slots, stream);
    track(history, stream);
    track(active, stream);
    track(map, stream);
    track(mult, stream);
    track(primes, stream);
    track(offsets, stream);
    track(out, stream);
    dsv41_engram_hash_launch(stream.stream(), u32(blocks), ptr(ids), ptr(pos), ptr(cu), ptr(slots),
                             ptr(history), ptr(active), ptr(map), ptr(mult), ptr(primes),
                             ptr(offsets), ptr(out), rows, batch, history_rows, vocab, pad, image,
                             mm_shift, u32(layers), u32(history.size(1)), u32(heads),
                             u32(hash_columns));
}
void engram_commit(const at::Tensor &ids, const at::Tensor &cu, const at::Tensor &slots,
                   const at::Tensor &accepted, const at::Tensor &history) {
    const auto device = ids.device();
    check(ids, at::kInt, device);
    check(slots, at::kInt, device);
    check(history, at::kInt, device);
    TORCH_CHECK_VALUE(ids.dim() == 1 && slots.dim() == 1 && history.dim() == 2 &&
                          history.size(1) > 0 &&
                          history.size(1) <= int64_t(arch2201::kUbBytes / sizeof(int32_t)),
                      "invalid Engram commit row axes");
    check(cu, at::kInt, device, {slots.numel() + 1});
    check(accepted, at::kInt, device, {slots.numel()});
    disjoint({history}, {ids, cu, slots, accepted});
    if (!slots.numel())
        return;

    const int64_t rows = ids.numel();
    const int64_t batch = slots.numel();
    const int64_t history_rows = history.size(0);
    const int64_t blocks = core_count(ids.get_device());

    c10_npu::NPUGuard guard(ids.device());
    auto stream = c10_npu::getCurrentNPUStream(ids.get_device());
    track(ids, stream);
    track(cu, stream);
    track(slots, stream);
    track(accepted, stream);
    track(history, stream);
    dsv41_engram_commit_launch(stream.stream(), u32(blocks), ptr(ids), ptr(cu), ptr(slots),
                               ptr(accepted), ptr(history), rows, batch, history_rows,
                               u32(history.size(1)));
}
void engram_gate(const at::Tensor &hidden, const at::Tensor &kv, const at::Tensor &qw,
                 const at::Tensor &kw, const at::Tensor &active, const at::Tensor &output) {
    const auto device = hidden.device();
    check(hidden, at::kBFloat16, device);
    TORCH_CHECK_VALUE(hidden.dim() == 3 && hidden.size(1) > 0 && hidden.size(2) >= 64 &&
                          hidden.size(2) % engram_ops::kFp32VectorElements == 0 &&
                          hidden.size(2) <=
                              int64_t((arch2201::kUbBytes - engram_ops::GateLayout::kStatBytes) /
                                      engram_ops::GateLayout::kBytesPerColumn),
                      "invalid Engram gate geometry/UB budget");
    check(kv, at::kBFloat16, device, {hidden.size(0), (hidden.size(1) + 1) * hidden.size(2)});
    check(qw, at::kBFloat16, device, {hidden.size(1), hidden.size(2)});
    check(kw, at::kBFloat16, device, qw.sizes());
    check(active, at::kInt, device, {hidden.size(0)});
    check(output, at::kBFloat16, device, hidden.sizes());
    disjoint({output}, {hidden, kv, qw, kw, active});
    if (!hidden.size(0))
        return;

    const int64_t tokens = hidden.size(0);
    const int64_t D = hidden.size(2);
    const int64_t branches = hidden.size(1);
    const double inv_sqrt_hidden = 1.0 / std::sqrt(double(D));
    const int64_t blocks = core_count(hidden.get_device());

    c10_npu::NPUGuard guard(hidden.device());
    auto stream = c10_npu::getCurrentNPUStream(hidden.get_device());
    track(hidden, stream);
    track(kv, stream);
    track(qw, stream);
    track(kw, stream);
    track(active, stream);
    track(output, stream);
    dsv41_engram_gate_launch(stream.stream(), u32(blocks), ptr(hidden), ptr(kv), ptr(qw), ptr(kw),
                             ptr(active), ptr(output), tokens, D, branches, f32(inv_sqrt_hidden));
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("prepare_engram_hash(Tensor ids, Tensor pos, Tensor cu, Tensor slots, Tensor history, "
          "Tensor "
          "active, Tensor map, Tensor mult, Tensor primes, Tensor offsets, Tensor out, int "
          "pad, int image, int mm_shift) -> ()",
          [](const at::Tensor &ids, const at::Tensor &pos, const at::Tensor &cu,
             const at::Tensor &slots, const at::Tensor &history, const at::Tensor &active,
             const at::Tensor &map, const at::Tensor &mult, const at::Tensor &primes,
             const at::Tensor &offsets, const at::Tensor &out, int64_t pad, int64_t image,
             int64_t mm_shift) {
              native_jit::PrepareOnlyScope prepare;
              engram_hash(ids, pos, cu, slots, history, active, map, mult, primes, offsets, out,
                          pad, image, mm_shift);
          });
    m.def("prepare_engram_commit(Tensor ids, Tensor cu, Tensor slots, Tensor accepted, Tensor "
          "history) -> ()",
          [](const at::Tensor &ids, const at::Tensor &cu, const at::Tensor &slots,
             const at::Tensor &accepted, const at::Tensor &history) {
              native_jit::PrepareOnlyScope prepare;
              engram_commit(ids, cu, slots, accepted, history);
          });
    m.def("prepare_engram_gate(Tensor hidden, Tensor kv, Tensor qw, Tensor kw, Tensor active, "
          "Tensor out) -> ()",
          [](const at::Tensor &hidden, const at::Tensor &kv, const at::Tensor &qw,
             const at::Tensor &kw, const at::Tensor &active, const at::Tensor &out) {
              native_jit::PrepareOnlyScope prepare;
              engram_gate(hidden, kv, qw, kw, active, out);
          });
    m.def("engram_hash(Tensor ids, Tensor pos, Tensor cu, Tensor slots, Tensor history, Tensor "
          "active, Tensor map, Tensor mult, Tensor primes, Tensor offsets, Tensor(a!) out, int "
          "pad, int image, int mm_shift) -> ()",
          &engram_hash);
    m.def("engram_commit(Tensor ids, Tensor cu, Tensor slots, Tensor accepted, Tensor(a!) history) "
          "-> ()",
          &engram_commit);
    m.def("engram_gate(Tensor hidden, Tensor kv, Tensor qw, Tensor kw, Tensor active, Tensor(a!) "
          "output) -> ()",
          &engram_gate);
}
} // namespace native_host
