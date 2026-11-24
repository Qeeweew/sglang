#include "torch.h"
#include "launch.h"
#include "jit.h"
#include "layout.h"
#include "workspace.h"

namespace native_host {
namespace {
using namespace qli;
int64_t query_tile_rows(int64_t rows, int64_t cores, int64_t heads) {
    TORCH_CHECK_VALUE(rows >= 0 && cores > 0 && heads > 0 && heads <= kHeadCapacity,
                      "invalid QLI scheduling extent");
    return plan_query_rows(rows, cores, heads);
}
void validate_geometry(int64_t heads, int64_t depth) {
    TORCH_CHECK_VALUE(heads > 0 && heads % kCubeRows == 0 && heads <= kHeadCapacity,
                      "QLI heads exceed the QK half-L0C budget or MMAD alignment");
    TORCH_CHECK_VALUE(depth > 0 && depth % kInt8DepthAlignment == 0 && depth <= kDepthCapacity,
                      "QLI depth exceeds the key L0B budget or INT8 C0 alignment");
}
int64_t qli_cores(int64_t device, int64_t requested) {
    const auto available = core_count(device, true);
    const auto cores = requested ? requested : available;
    TORCH_CHECK_VALUE(cores > 0 && cores <= available, "invalid QLI core count");
    return cores;
}
int64_t key_splits(int64_t rows, int64_t cores, int64_t heads, int64_t key_capacity) {
    TORCH_CHECK_VALUE(rows >= 0 && cores > 0 && heads > 0 && heads <= kHeadCapacity &&
                          key_capacity >= 0,
                      "invalid QLI scheduling extent");
    return plan_key_splits(rows, cores, heads, key_capacity);
}
int64_t workspace_size(int64_t device, int64_t requested, int64_t heads, int64_t rows, int64_t topk,
                       int64_t mode, int64_t key_capacity) {
    c10_npu::NPUGuard guard(at::Device(c10::DeviceType::PrivateUse1, device));
    TORCH_CHECK_VALUE(heads > 0 && heads <= kHeadCapacity && heads % kCubeRows == 0,
                      "QLI heads exceed pipeline capacity or Cube alignment");
    const auto cores = qli_cores(device, requested);
    Config c{};
    c.cores = u32(cores);
    c.heads = u32(heads);
    c.rows = u32(rows);
    c.topk = u32(topk);
    c.candidate_mode = u32(mode);
    c.query_tile_rows = u32(query_tile_rows(rows, cores, heads));
    c.key_splits = u32(key_splits(rows, cores, heads, key_capacity));
    return workspace_bytes(c);
}
void run_qli(const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, const at::Tensor &qs,
             const at::Tensor &ks, const at::Tensor &cu, const at::Tensor &lengths,
             const at::Tensor &residual, const at::Tensor &pages,
             const std::optional<at::Tensor> &candidates, const at::Tensor &indices,
             const std::optional<at::Tensor> &candidate_out,
             const std::optional<at::Tensor> &workspace,
             int64_t ratio, int64_t mode, int64_t max_q, int64_t max_k, int64_t max_page,
             int64_t requested, const std::optional<at::Tensor> &positions) {
    const auto device = q.device();
    check(q, at::kChar, device);
    TORCH_CHECK_VALUE(q.dim() == 3 && q.size(1) > 0 && q.size(1) <= kHeadCapacity &&
                          q.size(1) % kCubeRows == 0 && q.size(2) > 0 &&
                          q.size(2) <= kDepthCapacity && q.size(2) % kInt8DepthAlignment == 0,
                      "Q requires INT8 [rows,heads,depth] within arch22 slot capacity/alignment");
    const auto rows = q.size(0), batch = lengths.numel();
    const auto heads = q.size(1), depth = q.size(2);
    check(w, at::kHalf, device, {rows, heads});
    check(qs, at::kHalf, device, {rows, heads});
    check(indices, at::kInt, device);
    TORCH_CHECK_VALUE(indices.dim() == 2 && indices.size(0) == rows && indices.size(1) > 0 &&
                          indices.size(1) <= kKeyTileCapacity &&
                          indices.size(1) % kCandidateBlockTokens == 0,
                      "QLI topk must fit the rank slot and DMA alignment");
    const auto topk = indices.size(1);
    check(cu, at::kInt, device, {batch + 1});
    check(lengths, at::kInt, device, {batch});
    check(residual, at::kInt, device, {batch});
    check(pages, at::kInt, device);
    TORCH_CHECK_VALUE(pages.dim() == 2 && pages.size(0) == batch && pages.size(1) > 0,
                      "invalid QLI page table");
    TORCH_CHECK_VALUE(k.device() == device && k.scalar_type() == at::kChar && k.dim() == 3 &&
                          k.size(2) == depth && k.size(1) >= kCubeRows &&
                          k.size(1) % kCubeRows == 0 && k.stride(2) == 1 && k.stride(1) == depth &&
                          k.stride(0) >= k.size(1) * depth && k.size(0) > max_page,
                      "invalid paged K layout or physical page bound");
    TORCH_CHECK_VALUE(ks.device() == device && ks.scalar_type() == at::kHalf && ks.dim() == 2 &&
                          ks.size(0) == k.size(0) && ks.size(1) == k.size(1) && ks.stride(1) == 1 &&
                          ks.stride(0) >= k.size(1),
                      "invalid K scale layout");
    if (max_q == -1)
        max_q = rows;
    if (max_k == -1)
        max_k = pages.size(1) * k.size(1);
    TORCH_CHECK_VALUE((ratio == 1 || ratio == 2) && mode >= 1 && mode <= 3 && max_q >= 0 &&
                          max_k >= 0,
                      "invalid QLI compression/candidate configuration");
    TORCH_CHECK_VALUE((mode == 2) == bool(candidates) && (mode == 1) == bool(candidate_out),
                      "invalid candidate mode");
    if (candidates)
        check(*candidates, at::kInt, device, {rows, kKeyTileCapacity});
    if (candidate_out)
        check(*candidate_out, at::kInt, device, {rows, kKeyTileCapacity});
    if (positions)
        check(*positions, at::kInt, device, {rows});
    c10_npu::NPUGuard guard(device);
    const auto cores = qli_cores(q.get_device(), requested);
    Config tiling{};
    tiling.rows = u32(rows);
    tiling.batch = u32(batch);
    tiling.heads = u32(heads);
    tiling.depth = u32(depth);
    tiling.page_rows = u32(k.size(1));
    tiling.page_columns = u32(pages.size(1));
    tiling.key_page_stride = u32(k.stride(0));
    tiling.scale_page_stride = u32(ks.stride(0));
    tiling.ratio = u32(ratio);
    tiling.topk = u32(topk);
    tiling.candidate_mode = u32(mode);
    tiling.cores = u32(cores);
    tiling.query_tile_rows = u32(query_tile_rows(rows, cores, heads));
    tiling.key_splits = u32(key_splits(rows, cores, heads, pages.size(1) * k.size(1)));
    const auto bytes = int64_t(workspace_bytes(tiling));
    if (workspace) {
        check(*workspace, at::kByte, device);
        TORCH_CHECK_VALUE(workspace->numel() >= bytes, "QLI workspace too small");
    }
    disjoint({indices, candidate_out, workspace},
             {q, k, w, qs, ks, cu, lengths, residual, pages, candidates, positions});
    if (!rows)
        return;
    auto stream = c10_npu::getCurrentNPUStream(q.get_device());
    const auto acl_stream = stream.stream();
    const LaunchWorkspace scratch(workspace, bytes, acl_stream);
    for (const auto &t : {q, k, w, qs, ks, cu, lengths, residual, pages, indices})
        record(t, stream);
    track(workspace, stream);
    track(candidates, stream);
    track(candidate_out, stream);
    track(positions, stream);
    dsv41_qli_launch(acl_stream, u32(cores), &tiling, ptr(q), ptr(k), ptr(w), ptr(qs), ptr(ks),
                     ptr(cu), ptr(lengths), ptr(residual), ptr(pages), ptr(candidates),
                     ptr(indices), ptr(candidate_out), scratch.data(), ptr(positions));
}
void prepare_run_qli(const at::Tensor &q, const at::Tensor &k, const at::Tensor &w,
                     const at::Tensor &qs, const at::Tensor &ks, const at::Tensor &cu,
                     const at::Tensor &lengths, const at::Tensor &residual, const at::Tensor &pages,
                     const std::optional<at::Tensor> &candidates, const at::Tensor &indices,
                     const std::optional<at::Tensor> &candidate_out,
                     const std::optional<at::Tensor> &scratch,
                     int64_t ratio, int64_t mode, int64_t max_q, int64_t max_k, int64_t max_page,
                     int64_t requested, const std::optional<at::Tensor> &positions) {
    native_jit::PrepareOnlyScope scope;
    run_qli(q, k, w, qs, ks, cu, lengths, residual, pages, candidates, indices, candidate_out,
            scratch, ratio, mode, max_q, max_k, max_page, requested, positions);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("qli_geometry(int heads, int depth) -> ()", &validate_geometry);
    m.def("qli_query_tile_rows(int rows, int cores, int heads=32) -> int", &query_tile_rows);
    m.def("qli_key_splits(int rows, int cores, int heads, int key_capacity) -> int", &key_splits);
    m.def("qli_workspace_bytes(int device, int cores, int heads=32, int rows=0, int topk=512, int "
          "mode=3, int key_capacity=0) -> int",
          &workspace_size);
    m.def("qli(Tensor q, Tensor k, Tensor w, Tensor qs, Tensor ks, Tensor cu, Tensor lengths, "
          "Tensor residual, Tensor pages, Tensor? candidates, Tensor(a!) indices, Tensor(b!)? "
          "candidate_out, "
          "Tensor(c!)? workspace, int ratio, int mode, int max_q, int max_k, int max_page, int "
          "cores, Tensor? positions=None) -> ()",
          &run_qli);
    m.def("prepare_run_qli(Tensor q, Tensor k, Tensor w, Tensor qs, Tensor ks, Tensor cu, Tensor "
          "lengths, "
          "Tensor residual, Tensor pages, Tensor? candidates, Tensor(a!) indices, Tensor(b!)? "
          "candidate_out, "
          "Tensor(c!)? workspace, int ratio, int mode, int max_q, int max_k, int max_page, int "
          "cores, Tensor? positions=None) -> ()",
          &prepare_run_qli);
}
} // namespace native_host
