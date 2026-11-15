#include "torch.h"
#include "launch.h"
#include "jit.h"
#include "params.h"

namespace native_host {
namespace {
void compressor_prepare(const at::Tensor &kv, const std::optional<at::Tensor> &scores,
                        const at::Tensor &gamma, const at::Tensor &cu, const at::Tensor &requests,
                        const at::Tensor &pending, const at::Tensor &lengths,
                        const at::Tensor &pooled, const at::Tensor &latent,
                        const at::Tensor &positions, const at::Tensor &rope_positions,
                        const at::Tensor &starts, int64_t ratio) {
    const auto device = kv.device();
    TORCH_CHECK_VALUE(ratio == 1 || ratio == 2, "compressor ratio must be 1 or 2");
    check(kv, ratio == 1 ? at::kBFloat16 : at::kFloat, device);
    TORCH_CHECK_VALUE(kv.dim() == 2 && kv.size(1) > 0 &&
                          kv.size(1) <= compressor::kWidthCapacityElements &&
                          kv.size(1) % compressor::kWidthAlignmentElements == 0,
                      "projected KV width exceeds the aligned UB capacity");
    TORCH_CHECK_VALUE(bool(scores) == (ratio == 2),
                      "C1 has no score projection; C2 requires scores");
    if (scores)
        check(*scores, at::kFloat, device, kv.sizes());
    check(requests, at::kInt, device);
    TORCH_CHECK_VALUE(requests.dim() == 1, "request IDs must be a vector");
    check(cu, at::kInt, device, {requests.numel() + 1});
    check(gamma, at::kFloat, device, {kv.size(1)});
    check(lengths, at::kLong, device);
    TORCH_CHECK_VALUE(lengths.dim() == 1, "lengths must be a vector");
    check(pending, at::kFloat, device, {lengths.numel(), 2, kv.size(1)});
    check(pooled, at::kFloat, device, kv.sizes());
    check(latent, at::kBFloat16, device, kv.sizes());
    check(positions, at::kLong, device, {kv.size(0)});
    check(rope_positions, at::kLong, device, {kv.size(0)});
    check(starts, at::kLong, device, {requests.numel()});
    disjoint({kv, scores, gamma, cu, requests, pending, lengths, pooled, latent, positions,
              rope_positions, starts});
    TORCH_CHECK_VALUE(requests.numel() || !kv.size(0), "nonempty input needs at least one request");
    if (!requests.numel())
        return;

    const int64_t tokens = kv.size(0);
    const int64_t batches = requests.numel();
    const int64_t slots = lengths.numel();
    const int64_t blocks = core_count(kv.get_device());

    c10_npu::NPUGuard guard(kv.device());
    auto stream = c10_npu::getCurrentNPUStream(kv.get_device());
    track(kv, stream);
    track(scores, stream);
    track(gamma, stream);
    track(cu, stream);
    track(requests, stream);
    track(pending, stream);
    track(lengths, stream);
    track(pooled, stream);
    track(latent, stream);
    track(positions, stream);
    track(rope_positions, stream);
    track(starts, stream);
    dsv41_compressor_prepare_launch(stream.stream(), u32(blocks), ptr(kv), ptr(scores), ptr(gamma),
                                    ptr(cu), ptr(requests), ptr(pending), ptr(lengths), ptr(pooled),
                                    ptr(latent), ptr(positions), ptr(rope_positions), ptr(starts),
                                    tokens, batches, slots, ratio, kv.size(1));
}
void compressor_commit(const at::Tensor &kv, const std::optional<at::Tensor> &scores,
                       const at::Tensor &cu, const at::Tensor &requests, const at::Tensor &starts,
                       const at::Tensor &accepted, const at::Tensor &pending,
                       const at::Tensor &lengths, const at::Tensor &positions,
                       const at::Tensor &rope_positions, const at::Tensor &status, int64_t ratio) {
    const auto device = kv.device();
    TORCH_CHECK_VALUE(ratio == 1 || ratio == 2, "compressor ratio must be 1 or 2");
    check(kv, ratio == 1 ? at::kBFloat16 : at::kFloat, device);
    TORCH_CHECK_VALUE(
        kv.dim() == 2 && kv.size(1) > 0 && kv.size(1) <= compressor::kWidthCapacityElements &&
            kv.size(1) % compressor::kWidthAlignmentElements == 0 && bool(scores) == (ratio == 2),
        "invalid projected KV/scores");
    if (scores)
        check(*scores, at::kFloat, device, kv.sizes());
    check(requests, at::kInt, device);
    TORCH_CHECK_VALUE(requests.dim() == 1, "request IDs must be a vector");
    check(cu, at::kInt, device, {requests.numel() + 1});
    check(starts, at::kLong, device, {requests.numel()});
    check(accepted, at::kInt, device, {requests.numel()});
    check(lengths, at::kLong, device);
    TORCH_CHECK_VALUE(lengths.dim() == 1, "lengths must be a vector");
    check(pending, at::kFloat, device, {lengths.numel(), 2, kv.size(1)});
    check(positions, at::kLong, device, {kv.size(0)});
    check(rope_positions, at::kLong, device, {kv.size(0)});
    check(status, at::kInt, device, {requests.numel()});
    disjoint({kv, scores, cu, requests, starts, accepted, pending, lengths, positions,
              rope_positions, status});
    if (!requests.numel())
        return;

    const int64_t batches = requests.numel();
    const int64_t slots = lengths.numel();
    const int64_t blocks = core_count(kv.get_device());

    c10_npu::NPUGuard guard(kv.device());
    auto stream = c10_npu::getCurrentNPUStream(kv.get_device());
    track(kv, stream);
    track(scores, stream);
    track(cu, stream);
    track(requests, stream);
    track(starts, stream);
    track(accepted, stream);
    track(pending, stream);
    track(lengths, stream);
    track(positions, stream);
    track(rope_positions, stream);
    track(status, stream);
    dsv41_compressor_commit_launch(stream.stream(), u32(blocks), ptr(kv), ptr(scores), ptr(cu),
                                   ptr(requests), ptr(starts), ptr(accepted), ptr(pending),
                                   ptr(lengths), ptr(positions), ptr(rope_positions), ptr(status),
                                   batches, slots, ratio, kv.size(1));
}
void prepare_compressor_prepare(const at::Tensor &kv, const std::optional<at::Tensor> &scores,
                                const at::Tensor &gamma, const at::Tensor &cu,
                                const at::Tensor &requests, const at::Tensor &pending,
                                const at::Tensor &lengths, const at::Tensor &pooled,
                                const at::Tensor &latent, const at::Tensor &positions,
                                const at::Tensor &rope_positions, const at::Tensor &starts,
                                int64_t ratio) {
    native_jit::PrepareOnlyScope scope;
    compressor_prepare(kv, scores, gamma, cu, requests, pending, lengths, pooled, latent, positions,
                       rope_positions, starts, ratio);
}
void prepare_compressor_commit(const at::Tensor &kv, const std::optional<at::Tensor> &scores,
                               const at::Tensor &cu, const at::Tensor &requests,
                               const at::Tensor &starts, const at::Tensor &accepted,
                               const at::Tensor &pending, const at::Tensor &lengths,
                               const at::Tensor &positions, const at::Tensor &rope_positions,
                               const at::Tensor &status, int64_t ratio) {
    native_jit::PrepareOnlyScope scope;
    compressor_commit(kv, scores, cu, requests, starts, accepted, pending, lengths, positions,
                      rope_positions, status, ratio);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("compressor_prepare(Tensor kv, Tensor? scores, Tensor gamma, Tensor cu, Tensor requests, "
          "Tensor pending, Tensor lengths, Tensor(a!) pooled, Tensor(a!) latent, Tensor(a!) "
          "positions, Tensor(a!) rope_positions, Tensor(a!) starts, int ratio) -> ()",
          &compressor_prepare);
    m.def("prepare_compressor_prepare(Tensor kv, Tensor? scores, Tensor gamma, Tensor cu, Tensor "
          "requests, "
          "Tensor pending, Tensor lengths, Tensor(a!) pooled, Tensor(a!) latent, Tensor(a!) "
          "positions, Tensor(a!) rope_positions, Tensor(a!) starts, int ratio) -> ()",
          &prepare_compressor_prepare);
    m.def("compressor_commit(Tensor kv, Tensor? scores, Tensor cu, Tensor requests, Tensor starts, "
          "Tensor accepted, Tensor(a!) pending, Tensor(a!) lengths, Tensor(a!) positions, "
          "Tensor(a!) rope_positions, Tensor(a!) status, int ratio) -> ()",
          &compressor_commit);
    m.def("prepare_compressor_commit(Tensor kv, Tensor? scores, Tensor cu, Tensor requests, Tensor "
          "starts, "
          "Tensor accepted, Tensor(a!) pending, Tensor(a!) lengths, Tensor(a!) positions, "
          "Tensor(a!) rope_positions, Tensor(a!) status, int ratio) -> ()",
          &prepare_compressor_commit);
}
} // namespace native_host
