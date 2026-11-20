#include "torch.h"
#include "launch.h"
#include "jit.h"
#include "workspace.h"
#include <cmath>

namespace native_host {
attention::Config attention_geometry(int64_t tokens, int64_t heads, int64_t batch, int64_t columns,
                                     int64_t compressed_columns, int64_t capacity, int64_t ratio,
                                     int64_t mode, int64_t cores, bool ring, int64_t width,
                                     int64_t page_rows, int64_t window_rows,
                                     int64_t selected_rows) {
    TORCH_CHECK_VALUE(
        heads > 0 && heads <= attention::kHeadCapacity &&
            heads % attention::kMatrixFractalRows == 0 && width > 0 &&
            width <= attention::kWidthCapacity && width % attention::kMatrixFractalRows == 0 &&
            tokens >= 0 && tokens <= INT32_MAX && batch > 0 && batch <= INT32_MAX && columns > 0 &&
            columns <= INT32_MAX && compressed_columns >= 0 && compressed_columns <= INT32_MAX &&
            capacity > 0 && capacity <= attention::kSequenceCapacity &&
            capacity % attention::kDotRows == 0 && page_rows >= attention::kMatrixFractalRows &&
            page_rows <= INT32_MAX && page_rows % attention::kMatrixFractalRows == 0 &&
            window_rows > 0 && window_rows <= capacity && (ratio == 1 || ratio == 2) && mode >= 0 &&
            mode <= 2 && cores > 0,
        "attention geometry exceeds aligned allocation capacities");
    TORCH_CHECK_VALUE(
        (mode != 0 || capacity == ((window_rows + attention::kDotRows - 1) / attention::kDotRows) *
                                      attention::kDotRows) &&
            (mode != 1 || (window_rows <= attention::kSparseWindowCapacity && selected_rows > 0 &&
                           selected_rows <= attention::kSelectedCapacity &&
                           selected_rows % attention::kDotRows == 0 &&
                           capacity == attention::kSparseWindowCapacity + selected_rows &&
                           compressed_columns > 0)) &&
            (!ring || (mode == 1 && batch == 1)),
        "invalid attention mode/window/selected capacity");
    int tile_rows = heads;
    // A single query has no next Cube task to overlap the mandatory tiny-MMAD
    // fence. Use the next legal M extent; this extent is derived from the bound geometry.
    if (mode == 0 && tokens <= cores &&
        heads * attention::kDotRows / attention::kMatrixFractalElements <
            arch2201::kUnfencedMmadFractals)
        tile_rows = attention::kSmallQueryTileRows;
    int compressed_page_rows = page_rows / ratio;
    int shift = 0;
    while ((int64_t(1) << shift) < compressed_page_rows)
        ++shift;
    bool power_of_two = (int64_t(1) << shift) == compressed_page_rows;
    return {int(tokens),
            int(heads),
            int(width),
            int(batch),
            int(columns),
            int(compressed_columns),
            int(page_rows),
            compressed_page_rows,
            int(window_rows),
            int(selected_rows),
            int(capacity),
            int(ratio),
            int(mode),
            int(cores),
            tile_rows,
            power_of_two ? shift : -1,
            int32_t(power_of_two ? shift + (ratio - 1) : -1),
            ring,
            false,
            false,
            1.0f / std::sqrt(float(width))};
}
namespace {
int64_t selected_cores(int64_t device, int64_t requested) {
    const auto available = core_count(device, true);
    TORCH_CHECK_VALUE(requested >= 0 && requested <= available,
                      "launch core count exceeds device capacity");
    return requested ? requested : available;
}
int64_t workspace(int64_t tokens, int64_t heads, int64_t batch, int64_t columns,
                  int64_t compressed_columns, int64_t capacity, int64_t ratio, int64_t mode,
                  int64_t device, bool ring, int64_t width, int64_t page_rows, int64_t window_rows,
                  int64_t selected_rows, int64_t cores) {
    c10_npu::NPUGuard guard(at::Device(c10::DeviceType::PrivateUse1, device));
    return attention::workspace_bytes(attention_geometry(
        tokens, heads, batch, columns, compressed_columns, capacity, ratio, mode,
        selected_cores(device, cores), ring, width, page_rows, window_rows, selected_rows));
}
void run(const at::Tensor &q, const at::Tensor &kv, const std::optional<at::Tensor> &compressed,
         const std::optional<at::Tensor> &indices, const at::Tensor &table,
         const std::optional<at::Tensor> &compressed_table, const at::Tensor &cu,
         const at::Tensor &lengths, const std::optional<at::Tensor> &compressed_lengths,
         const std::optional<at::Tensor> &residual, const std::optional<at::Tensor> &window,
         const at::Tensor &sinks, const at::Tensor &out, const std::optional<at::Tensor> &workspace,
         int64_t capacity, int64_t ratio, int64_t mode, bool ring, int64_t window_rows,
         int64_t cores) {
    const auto device = q.device();
    TORCH_CHECK_VALUE(q.dim() == 3 && table.dim() == 2 && table.size(0) > 0 && table.size(1) > 0 &&
                          kv.dim() == 4 && kv.size(0) > 0 && kv.size(2) == 1,
                      "requires Q[T,H,D], paged KV[P,page_rows,1,D] and positive page tables");
    c10_npu::NPUGuard guard(device);
    const auto selected_rows = mode == 1 && indices && indices->dim() == 3 ? indices->size(2) : 0;
    auto c = attention_geometry(
        q.size(0), q.size(1), table.size(0), table.size(1),
        compressed_table && compressed_table->dim() == 2 ? compressed_table->size(1) : 0, capacity,
        ratio, mode, selected_cores(device.index(), cores), ring, q.size(2), kv.size(1),
        window_rows, selected_rows);
    check(q, at::kBFloat16, device);
    check(kv, at::kBFloat16, device, {kv.size(0), c.page_rows, 1, c.width});
    check(table, at::kInt, device, {c.batch, c.columns});
    check(cu, at::kInt, device, {int64_t(c.batch) + 1});
    check(lengths, at::kInt, device, {c.batch});
    check(sinks, at::kFloat, device, {c.heads});
    check(out, at::kBFloat16, device, q.sizes());
    TORCH_CHECK_VALUE(mode == 2 || window, "causal attention requires window lengths");
    if (window)
        check(*window, at::kInt, device, {c.tokens, 1});
    if (mode == 1) {
        TORCH_CHECK_VALUE(compressed && indices && compressed_table && compressed_lengths &&
                              residual && compressed->dim() == 4 && compressed->size(0) > 0,
                          "missing compressed inputs");
        check(*compressed, at::kBFloat16, device,
              {compressed->size(0), c.compressed_page_rows, 1, c.width});
        check(*indices, at::kInt, device, {c.tokens, 1, c.selected_rows});
        check(*compressed_table, at::kInt, device, {c.batch, c.compressed_columns});
        check(*compressed_lengths, at::kInt, device, {c.batch});
        check(*residual, at::kInt, device, {c.batch});
        c.coordinate32 =
            c.coordinate_shift > 0 && compressed->size(0) <= INT32_MAX / c.compressed_page_rows;
        c.coordinate_elements32 = compressed->numel() <= INT32_MAX;
    }
    const auto bytes = attention::workspace_bytes(c);
    if (workspace) {
        check(*workspace, at::kByte, device);
        TORCH_CHECK_VALUE(workspace->numel() >= bytes, "attention workspace too small");
    }
    disjoint({out, workspace}, {q, kv, compressed, indices, table, compressed_table, cu, lengths,
                              compressed_lengths, residual, window, sinks});
    if (!c.tokens)
        return;
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    const auto acl_stream = stream.stream();
    const LaunchWorkspace scratch(workspace, bytes, acl_stream);
    for (const auto &t : {q, kv, table, cu, lengths, sinks, out})
        record(t, stream);
    track(workspace, stream);
    for (const auto &t :
         {compressed, indices, compressed_table, compressed_lengths, residual, window})
        track(t, stream);
    attention::Arguments a{{uintptr_t(ptr(q)), uintptr_t(ptr(kv)), uintptr_t(ptr(compressed)),
                            uintptr_t(ptr(indices)), uintptr_t(ptr(table)),
                            uintptr_t(ptr(compressed_table)), uintptr_t(ptr(cu)),
                            uintptr_t(ptr(lengths)), uintptr_t(ptr(compressed_lengths)),
                            uintptr_t(ptr(residual)), uintptr_t(ptr(window)), uintptr_t(ptr(sinks)),
                            uintptr_t(ptr(out)), uintptr_t(scratch.data())},
                           c};
    dsv41_attention_launch(acl_stream, a);
}

} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("attention_workspace(int tokens, int heads, int batch, int columns, int "
          "compressed_columns, "
          "int capacity, int ratio, int mode, int device, bool ring, int width=512, int "
          "page_rows=128, "
          "int window_rows=128, int selected_rows=512, int cores=0) -> int",
          &workspace);
    const auto schema = "(Tensor q, Tensor kv, Tensor? compressed, Tensor? indices, Tensor table, "
                        "Tensor? compressed_table, Tensor cu, Tensor lengths, Tensor? "
                        "compressed_lengths, Tensor? residual, "
                        "Tensor? window, Tensor sinks, Tensor(a!) out, Tensor(b!)? workspace, int "
                        "capacity, int ratio, "
                        "int mode, bool ring, int window_rows=128, int cores=0) -> ()";
    m.def((std::string("attention") + schema).c_str(), &run);
    m.def((std::string("prepare_attention") + schema).c_str(),
          [](const at::Tensor &q, const at::Tensor &kv, const std::optional<at::Tensor> &compressed,
             const std::optional<at::Tensor> &indices, const at::Tensor &table,
             const std::optional<at::Tensor> &compressed_table, const at::Tensor &cu,
             const at::Tensor &lengths, const std::optional<at::Tensor> &compressed_lengths,
             const std::optional<at::Tensor> &residual, const std::optional<at::Tensor> &window,
             const at::Tensor &sinks, const at::Tensor &out, const std::optional<at::Tensor> &scratch,
             int64_t capacity, int64_t ratio, int64_t mode, bool ring, int64_t window_rows,
             int64_t cores) {
              native_jit::PrepareOnlyScope scope;
              run(q, kv, compressed, indices, table, compressed_table, cu, lengths,
                  compressed_lengths, residual, window, sinks, out, scratch, capacity, ratio, mode,
                  ring, window_rows, cores);
          });
}
} // namespace native_host
