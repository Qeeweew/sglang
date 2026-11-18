#include "launch.h"
#include "torch.h"
#include "jit.h"
#include "workspace.h"
namespace native_host {
namespace {
using namespace mhc;
struct Shape {
    int64_t rows;
    uint32_t branches, width;
};
Shape residual(const at::Tensor &r) {
    check(r, at::kBFloat16, r.device());
    TORCH_CHECK_VALUE(r.dim() == 3, "residual requires BF16 [rows,branches,width]");
    const uint32_t b = u32(r.size(1)), h = u32(r.size(2));
    TORCH_CHECK_VALUE(b > 0 && b <= kBranchCapacity && h > 0 && h % (2 * kRepeatFloats) == 0 &&
                          h <= 2 * kRepeatFloats * kRepeatFloats,
                      "mHC branches must fit one FP32 block; width must be positive,128-aligned "
                      "and <=8192 (two r64 reduction panels)");
    TORCH_CHECK_VALUE(CombineLayout(b, h, kRepeatFloats).bytes <= arch2201::kUbBytes &&
                          PostLayout(b, h, kRepeatFloats).bytes <= arch2201::kUbBytes,
                      "mHC state does not fit arch2201 UB");
    return {r.size(0), b, h};
}
ProjectParams plan(const Shape &s, int device) {
    auto p = project_plan(s.rows, s.branches, s.width, u32(core_count(device, true)),
                          u32(core_count(device)));
    TORCH_CHECK_VALUE(p.panel_k, "mHC projection exceeds arch2201 storage");
    if (s.rows > kStatsRowCapacity) {
        TORCH_CHECK_VALUE(p.l1_tile_k &&
                              uint64_t(kSlots) * p.tile_rows * p.l1_tile_k * kFp32Bytes +
                                      uint64_t(p.projection_stride) * p.panel_k * kFp32Bytes <=
                                  arch2201::kL1Bytes,
                          "mHC L1 budget exceeded");
    }
    return p;
}
std::vector<int64_t> geometry(int64_t rows, int64_t branches, int64_t width, int64_t device) {
    TORCH_CHECK_VALUE(rows >= 0 && branches > 0 && branches <= kBranchCapacity && width > 0 &&
                          width % kHiddenWidthAlignment == 0 && width <= kHiddenWidthCapacity,
                      "invalid mHC dimensions");
    auto p = plan({rows, u32(branches), u32(width)}, device);
    return {p.tile_rows,   int64_t(p.padded_rows), p.splits,      p.panel_k,   p.projection_stride,
            p.cube_groups, p.vector_groups,        p.panel_count, p.l1_tile_k, p.row_groups};
}
StateParams state(const Shape &s, uint32_t tile) {
    return {uint64_t(s.rows), s.width, s.branches, tile, (s.width + tile - 1) / tile, 0, 0};
}
void project(const at::Tensor &r, const at::Tensor &fn, const at::Tensor &projection,
             const at::Tensor &squares, const std::optional<at::Tensor> &inputs,
             const std::optional<at::Tensor> &products, int64_t bm, bool vector) {
    const auto s = residual(r);
    auto p = plan(s, r.get_device());
    auto device = r.device();
    if (vector) {
        p.panel_k = kSmallProjectionPanelCapacity;
        p.panel_count = (p.input_width + p.panel_k - 1) / p.panel_k;
        p.splits = p.panel_count;
        p.vector_groups = core_count(r.get_device());
    }
    TORCH_CHECK_VALUE(bm == p.tile_rows, "mHC projection geometry differs from runtime plan");
    check(fn, at::kFloat, device, {p.projection_columns, p.input_width});
    check(projection, at::kFloat, device, {p.splits, int64_t(p.padded_rows), p.projection_stride});
    check(squares, at::kFloat, device, {p.splits, int64_t(p.padded_rows)});
    const std::array<int64_t, 4> input_shape{p.cube_groups, kSlots, p.tile_rows, p.panel_k};
    const std::array<int64_t, 4> product_shape{p.cube_groups, kSlots, p.tile_rows,
                                               p.projection_stride};
    if (!vector) {
        TORCH_CHECK_VALUE(p.input_width <= UINT16_MAX,
                          "mixed mHC Fn ND2NZ source stride exceeds uint16");
        if (inputs)
            check(*inputs, at::kFloat, device, input_shape);
        if (products)
            check(*products, at::kFloat, device, product_shape);
    }
    disjoint({projection, squares, inputs, products}, {r, fn});
    if (!s.rows)
        return;
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(r.get_device());
    const auto acl_stream = stream.stream();
    const size_t slot_bytes = vector ? 0 : size_t(p.cube_groups) * kSlots * p.tile_rows * sizeof(float);
    const LaunchWorkspace input_ring(inputs, slot_bytes * p.panel_k, acl_stream);
    const LaunchWorkspace product_ring(products, slot_bytes * p.projection_stride, acl_stream);
    for (auto &t : {r, fn, projection, squares})
        track(t, stream);
    track(inputs, stream);
    track(products, stream);
    dsv41_mhc_project_launch(acl_stream, vector ? p.vector_groups : p.cube_groups, ptr(r), ptr(fn),
                             ptr(projection), ptr(squares), input_ring.data(), product_ring.data(),
                             p, vector);
}
StatsParams stats_check(const at::Tensor &p, const at::Tensor &s, const at::Tensor &scale,
                        const at::Tensor &base, const at::Tensor &pre, const at::Tensor &post,
                        const at::Tensor &comb, int64_t width, double epsilon) {
    auto device = p.device();
    check(p, at::kFloat, device);
    check(pre, at::kFloat, device);
    TORCH_CHECK_VALUE(pre.dim() == 2 && pre.size(1) > 0 && pre.size(1) <= kBranchCapacity &&
                          width > 0 && width % kHiddenWidthAlignment == 0 &&
                          width <= kHiddenWidthCapacity,
                      "invalid mHC statistics dimensions");
    const auto rows = pre.size(0);
    const uint32_t b = u32(pre.size(1)), columns = b * (b + 2),
                   stride = round_up(columns, kMatrixRows);
    TORCH_CHECK_VALUE(p.dim() == 3 && p.size(0) > 0 && p.size(1) >= rows && p.size(2) == stride,
                      "invalid mHC projection workspace");
    check(s, at::kFloat, device, {p.size(0), p.size(1)});
    check(scale, at::kFloat, device, {3});
    check(base, at::kFloat, device, {columns});
    check(post, at::kFloat, device, {rows, b});
    check(comb, at::kFloat, device, {rows, b, b});
    TORCH_CHECK_VALUE(std::isfinite(epsilon) && epsilon > 0,
                      "mHC epsilon must be positive and finite");
    disjoint({pre, post, comb}, {p, s, scale, base});
    const uint32_t r = rows >= kCubeRowCapacity ? kStatsRowCapacity : 1;
    TORCH_CHECK_VALUE(p.size(1) % r == 0, "mHC projection row padding must cover statistics tiles");
    return {uint64_t(rows),
            uint64_t(p.size(1)),
            b,
            u32(width),
            columns,
            stride,
            u32(p.size(0)),
            r,
            0,
            0,
            1.0f / (b * width),
            f32(epsilon)};
}
void stats(const at::Tensor &p, const at::Tensor &s, const at::Tensor &scale,
           const at::Tensor &base, const at::Tensor &pre, const at::Tensor &post,
           const at::Tensor &comb, int64_t width, double epsilon) {
    auto parameters = stats_check(p, s, scale, base, pre, post, comb, width, epsilon);
    if (!parameters.rows)
        return;
    c10_npu::NPUGuard guard(p.device());
    auto stream = c10_npu::getCurrentNPUStream(p.get_device());
    for (auto &t : {p, s, scale, base, pre, post, comb})
        track(t, stream);
    dsv41_mhc_stats_launch(stream.stream(),
                           u32(core_count(p.get_device())), ptr(p),
                           ptr(s), ptr(scale), ptr(base), ptr(pre), ptr(post), ptr(comb), nullptr,
                           nullptr, nullptr, parameters, false);
}
void stats_post(const at::Tensor &p, const at::Tensor &s, const at::Tensor &scale,
                const at::Tensor &base, const at::Tensor &pre, const at::Tensor &post,
                const at::Tensor &comb, double epsilon, const at::Tensor &x, const at::Tensor &r,
                const at::Tensor &out) {
    const auto shape = residual(r);
    auto parameters = stats_check(p, s, scale, base, pre, post, comb, shape.width, epsilon);
    TORCH_CHECK_VALUE(parameters.rows == uint64_t(shape.rows) &&
                          parameters.branches == shape.branches,
                      "mHC stats/post rows differ");
    check(x, at::kBFloat16, r.device(), {shape.rows, shape.width});
    check(out, at::kBFloat16, r.device(), r.sizes());
    disjoint({pre, post, comb, out}, {p, s, scale, base, x, r});
    if (!shape.rows)
        return;
    parameters.post_tile_columns =
        stats_post_tile(shape.rows, shape.branches, shape.width, u32(core_count(r.get_device())));
    parameters.post_column_tiles =
        (shape.width + parameters.post_tile_columns - 1) / parameters.post_tile_columns;
    c10_npu::NPUGuard guard(r.device());
    auto stream = c10_npu::getCurrentNPUStream(r.get_device());
    for (auto &t : {p, s, scale, base, pre, post, comb, x, r, out})
        track(t, stream);
    dsv41_mhc_stats_launch(stream.stream(),
                           u32(core_count(r.get_device())),
                           ptr(p), ptr(s), ptr(scale), ptr(base), ptr(pre), ptr(post), ptr(comb),
                           ptr(x), ptr(r), ptr(out), parameters, true);
}
void combine(const at::Tensor &r, const at::Tensor &pre, const std::optional<at::Tensor> &weight,
             const at::Tensor &out, bool initial) {
    const auto s = residual(r);
    auto device = r.device();
    check(pre, at::kFloat, device, {s.rows, s.branches});
    check(out, at::kBFloat16, device, {s.rows, s.width});
    if (weight)
        check(*weight, at::kFloat, device, {s.width});
    disjoint({out}, {r, pre, weight});
    if (!s.rows)
        return;
    uint32_t tile = combine_tile(s.branches, s.width);
    if (!initial && !weight) {
        tile = s.width;
        while (uint64_t(6 * s.branches + 12) * tile + 2 * kBankPhaseBytes > arch2201::kUbBytes)
            tile -= kRepeatFloats;
    }
    auto parameters = state(s, tile);
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(r.get_device());
    for (auto &t : {r, pre, out})
        track(t, stream);
    track(weight, stream);
    dsv41_mhc_combine_launch(stream.stream(),
                             u32(core_count(r.get_device())), ptr(r),
                             ptr(pre), ptr(weight), ptr(out), parameters, initial, bool(weight));
}
void post_call(const at::Tensor &x, const at::Tensor &r, const at::Tensor &post,
               const at::Tensor &comb, const at::Tensor &out) {
    const auto s = residual(r);
    auto device = r.device();
    check(x, at::kBFloat16, device, {s.rows, s.width});
    check(post, at::kFloat, device, {s.rows, s.branches});
    check(comb, at::kFloat, device, {s.rows, s.branches, s.branches});
    check(out, at::kBFloat16, device, r.sizes());
    disjoint({out}, {x, r, post, comb});
    if (!s.rows)
        return;
    auto parameters =
        state(s, post_tile(s.rows, s.branches, s.width, u32(core_count(r.get_device())), false));
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(r.get_device());
    for (auto &t : {x, r, post, comb, out})
        track(t, stream);
    dsv41_mhc_post_launch(
        stream.stream(),
        u32(core_count(r.get_device())),
        ptr(x), ptr(r), nullptr, ptr(post), ptr(comb), nullptr, ptr(out), nullptr, parameters,
        false);
}
void post_combine(const at::Tensor &x, const at::Tensor &r, const at::Tensor &pre,
                  const at::Tensor &post, const at::Tensor &comb, const at::Tensor &weight,
                  const at::Tensor &updated, const at::Tensor &out) {
    const auto s = residual(r);
    auto device = r.device();
    check(x, at::kBFloat16, device, {s.rows, s.width});
    check(pre, at::kFloat, device, {s.rows, s.branches});
    check(post, at::kFloat, device, {s.rows, s.branches});
    check(comb, at::kFloat, device, {s.rows, s.branches, s.branches});
    check(weight, at::kFloat, device, {s.width});
    check(updated, at::kBFloat16, device, r.sizes());
    check(out, at::kBFloat16, device, {s.rows, s.width});
    disjoint({updated, out}, {x, r, pre, post, comb, weight});
    if (!s.rows)
        return;
    auto parameters =
        state(s, post_tile(s.rows, s.branches, s.width, u32(core_count(r.get_device())), true));
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(r.get_device());
    for (auto &t : {x, r, pre, post, comb, weight, updated, out})
        track(t, stream);
    dsv41_mhc_post_launch(stream.stream(),
                          u32(core_count(r.get_device())), ptr(x),
                          ptr(r), ptr(pre), ptr(post), ptr(comb), ptr(weight), ptr(updated),
                          ptr(out), parameters, true);
}
void mean(const at::Tensor &r, const at::Tensor &out, int64_t column) {
    const auto s = residual(r);
    auto device = r.device();
    check(out, at::kBFloat16, device);
    TORCH_CHECK_VALUE(out.dim() == 2 && out.size(0) == s.rows && out.size(1) % s.width == 0 &&
                          column >= 0 && column < out.size(1) / s.width,
                      "invalid packed mHC capture column");
    disjoint({out}, {r});
    if (!s.rows)
        return;
    const uint32_t tile = mean_tile(s.branches, s.width);
    auto parameters = state(s, tile);
    parameters.column = u32(column);
    parameters.output_columns = u32(out.size(1) / s.width);
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(r.get_device());
    track(r, stream);
    track(out, stream);
    dsv41_mhc_mean_launch(
        stream.stream(),
        u32(core_count(r.get_device())),
        ptr(r), ptr(out), parameters);
}
// Offline preparation uses the same actual planners and launch selectors as tensor binding.
void precompile(const std::string &stage, int64_t rows, int64_t branches, int64_t width,
                int64_t output_columns, bool initial, bool normalize) {
    TORCH_CHECK_VALUE(rows > 0 && branches > 0 && branches <= kBranchCapacity && width > 0 &&
                          width % kHiddenWidthAlignment == 0 && width <= kHiddenWidthCapacity,
                      "invalid actual mHC geometry");
    const int device = c10_npu::current_device();
    const Shape shape{rows, u32(branches), u32(width)};
    const auto project_plan = plan(shape, device);
    const auto workers = u32(core_count(device));
    native_jit::PrepareOnlyScope scope;
    if (stage == "project" || stage == "project_vector") {
        auto selected = project_plan;
        const bool vector = stage == "project_vector";
        if (vector) {
            selected.panel_k = kSmallProjectionPanelCapacity;
            selected.panel_count = (selected.input_width + selected.panel_k - 1) / selected.panel_k;
            selected.splits = selected.panel_count;
            selected.vector_groups = workers;
        } else {
            TORCH_CHECK_VALUE(selected.input_width <= UINT16_MAX,
                              "mixed mHC Fn ND2NZ source stride exceeds uint16");
        }
        dsv41_mhc_project_launch(nullptr, vector ? selected.vector_groups : selected.cube_groups,
                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, selected,
                                 vector);
    } else if (stage == "stats" || stage == "stats_post") {
        const bool fused = stage == "stats_post";
        const uint32_t tile = fused ? stats_post_tile(rows, branches, width, workers) : 0;
        StatsParams selected{uint64_t(rows),
                             project_plan.padded_rows,
                             u32(branches),
                             u32(width),
                             project_plan.projection_columns,
                             project_plan.projection_stride,
                             project_plan.splits,
                             rows >= kCubeRowCapacity ? kStatsRowCapacity : 1,
                             tile,
                             tile ? (u32(width) + tile - 1) / tile : 0,
                             1.0f / (branches * width),
                             1e-20f};
        dsv41_mhc_stats_launch(nullptr, workers, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, selected, fused);
    } else if (stage == "combine" || stage == "collapse") {
        const bool collapse = stage == "collapse";
        uint32_t tile = combine_tile(branches, width);
        if (collapse) {
            tile = width;
            while (uint64_t(6 * branches + 12) * tile + 2 * kBankPhaseBytes > arch2201::kUbBytes)
                tile -= kRepeatFloats;
        }
        const auto selected = state(shape, tile);
        TORCH_CHECK_VALUE(collapse || initial || normalize,
                          "use collapse for unnormalized weighted state");
        dsv41_mhc_combine_launch(nullptr, workers, nullptr, nullptr,
                                 nullptr, nullptr, selected, collapse ? false : initial,
                                 collapse ? false : normalize);
    } else if (stage == "post" || stage == "post_combine") {
        const bool fused = stage == "post_combine";
        const auto selected = state(shape, post_tile(rows, branches, width, workers, fused));
        dsv41_mhc_post_launch(nullptr, workers, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, selected,
                              fused);
    } else {
        TORCH_CHECK_VALUE(stage == "mean" && output_columns > 0 && output_columns <= UINT32_MAX,
                          "supply an actual mHC stage and positive packed output columns");
        auto selected = state(shape, mean_tile(branches, width));
        selected.output_columns = u32(output_columns);
        dsv41_mhc_mean_launch(nullptr,
                              workers,
                              nullptr, nullptr, selected);
    }
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_mhc(str stage, int rows, int branches, int width, int output_columns=1, "
          "bool initial=False, bool normalize=True) -> ()",
          &precompile);
    m.def("mhc_geometry(int rows, int branches, int width, int device) -> int[]", &geometry);
    m.def("mhc_project(Tensor residual, Tensor fn, Tensor(a!) projection, Tensor(a!) squares, "
          "Tensor(a!)? inputs, Tensor(a!)? products, int bm, bool vector) -> ()",
          &project);
    m.def("prepare_mhc_project(Tensor residual, Tensor fn, Tensor(a!) projection, Tensor(a!) "
          "squares, "
          "Tensor(a!)? inputs, Tensor(a!)? products, int bm, bool vector) -> ()",
          [](const at::Tensor &r, const at::Tensor &fn, const at::Tensor &projection,
             const at::Tensor &squares, const std::optional<at::Tensor> &inputs,
             const std::optional<at::Tensor> &products, int64_t bm, bool vector) {
              native_jit::PrepareOnlyScope scope;
              project(r, fn, projection, squares, inputs, products, bm, vector);
          });
    m.def("mhc_stats(Tensor projection, Tensor squares, Tensor scale, Tensor base, Tensor(a!) pre, "
          "Tensor(a!) post, Tensor(a!) comb, int hidden_width, float epsilon) -> ()",
          &stats);
    m.def("prepare_mhc_stats(Tensor projection, Tensor squares, Tensor scale, Tensor base, "
          "Tensor(a!) pre, "
          "Tensor(a!) post, Tensor(a!) comb, int hidden_width, float epsilon) -> ()",
          [](const at::Tensor &p, const at::Tensor &s, const at::Tensor &scale,
             const at::Tensor &base, const at::Tensor &pre, const at::Tensor &post,
             const at::Tensor &comb, int64_t width, double epsilon) {
              native_jit::PrepareOnlyScope scope;
              stats(p, s, scale, base, pre, post, comb, width, epsilon);
          });
    m.def("mhc_stats_post(Tensor projection, Tensor squares, Tensor scale, Tensor base, Tensor(a!) "
          "pre, Tensor(a!) post, Tensor(a!) comb, float epsilon, Tensor x, Tensor residual, "
          "Tensor(a!) out) -> ()",
          &stats_post);
    m.def("prepare_mhc_stats_post(Tensor projection, Tensor squares, Tensor scale, Tensor base, "
          "Tensor(a!) "
          "pre, Tensor(a!) post, Tensor(a!) comb, float epsilon, Tensor x, Tensor residual, "
          "Tensor(a!) out) -> ()",
          [](const at::Tensor &p, const at::Tensor &s, const at::Tensor &scale,
             const at::Tensor &base, const at::Tensor &pre, const at::Tensor &post,
             const at::Tensor &comb, double epsilon, const at::Tensor &x, const at::Tensor &r,
             const at::Tensor &out) {
              native_jit::PrepareOnlyScope scope;
              stats_post(p, s, scale, base, pre, post, comb, epsilon, x, r, out);
          });
    m.def("mhc_combine(Tensor residual, Tensor pre, Tensor? weight, Tensor(a!) out, bool initial) "
          "-> ()",
          &combine);
    m.def("prepare_mhc_combine(Tensor residual, Tensor pre, Tensor? weight, Tensor(a!) out, bool "
          "initial) "
          "-> ()",
          [](const at::Tensor &r, const at::Tensor &pre, const std::optional<at::Tensor> &weight,
             const at::Tensor &out, bool initial) {
              native_jit::PrepareOnlyScope scope;
              combine(r, pre, weight, out, initial);
          });
    m.def("mhc_post(Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor(a!) out) -> ()",
          &post_call);
    m.def("prepare_mhc_post(Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor(a!) out) "
          "-> ()",
          [](const at::Tensor &x, const at::Tensor &r, const at::Tensor &post,
             const at::Tensor &comb, const at::Tensor &out) {
              native_jit::PrepareOnlyScope scope;
              post_call(x, r, post, comb, out);
          });
    m.def("mhc_post_combine(Tensor x, Tensor residual, Tensor pre, Tensor post, Tensor comb, "
          "Tensor weight, Tensor(a!) updated, Tensor(a!) out) -> ()",
          &post_combine);
    m.def(
        "prepare_mhc_post_combine(Tensor x, Tensor residual, Tensor pre, Tensor post, Tensor comb, "
        "Tensor weight, Tensor(a!) updated, Tensor(a!) out) -> ()",
        [](const at::Tensor &x, const at::Tensor &r, const at::Tensor &pre, const at::Tensor &post,
           const at::Tensor &comb, const at::Tensor &weight, const at::Tensor &updated,
           const at::Tensor &out) {
            native_jit::PrepareOnlyScope scope;
            post_combine(x, r, pre, post, comb, weight, updated, out);
        });
    m.def("mhc_mean(Tensor residual, Tensor(a!) packed, int column) -> ()", &mean);
    m.def("prepare_mhc_mean(Tensor residual, Tensor(a!) packed, int column) -> ()",
          [](const at::Tensor &r, const at::Tensor &out, int64_t column) {
              native_jit::PrepareOnlyScope scope;
              mean(r, out, column);
          });
}
} // namespace native_host
