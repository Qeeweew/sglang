#include "launch.h"
#include "actual_entry.h"
#include "torch.h"
#include <algorithm>
namespace {
using native_jit::Kernel;
using namespace mhc;
enum class Flow : uint32_t {
    ProjectSmall,
    ProjectMixed,
    StatsScalar,
    StatsBlock,
    StatsPost,
    Combine,
    Collapse,
    Mean,
    Post,
    PostCombine
};
using Entries = native_jit::ActualEntries<12>;
// Flags are built inside the cache miss creator, never on hot launch.
template <class Flags>
Kernel &actual_entry(const Entries::Key &key, const char *tag, const char *source, uint32_t workers,
                     Flags make_flags) {
    static thread_local Entries entries;
    return entries.get(key, [&] {
        auto flags = make_flags();
        flags.emplace_back("--cce-auto-sync=off");
        return std::make_unique<Kernel>(tag, source, flags, workers);
    });
}
void field(std::vector<std::string> &flags, const char *name, uint64_t value) {
    flags.emplace_back(std::string("-DMHC_") + name + "=" + std::to_string(value));
}
std::vector<std::string> common(uint32_t branches, uint32_t width,
                                uint32_t workers) {
    std::vector<std::string> flags;
    field(flags, "BRANCHES", branches);
    field(flags, "WIDTH", width);
    field(flags, "WORKERS", workers);
    return flags;
}
Kernel &project_entry(const ProjectParams &p, uint32_t workers, bool vector) {
    const Entries::Key key{uint64_t(vector ? Flow::ProjectSmall : Flow::ProjectMixed),
                           p.branches,
                           p.hidden_width,
                           p.tile_rows,
                           p.panel_k,
                           p.l1_tile_k,
                           p.splits,
                           p.row_groups,
                           workers};
    return actual_entry(key, vector ? "mhc_project_small" : "mhc_project",
                        vector ? "mhc/project_small.asc" : "mhc/project.asc", workers, [&] {
                            auto flags = common(p.branches, p.hidden_width, workers);
                            flags.emplace_back("-DMHC_PROJECT_PLAN=1");
                            field(flags, "TILE_ROWS", p.tile_rows);
                            field(flags, "PANEL_K", p.panel_k);
                            field(flags, "L1_TILE_K", p.l1_tile_k);
                            field(flags, "SPLITS", p.splits);
                            field(flags, "ROW_GROUPS", p.row_groups);
                            return flags;
                        });
}
Kernel &stats_entry(const StatsParams &p, uint32_t workers, bool fused) {
    const Entries::Key key{uint64_t(fused              ? Flow::StatsPost
                                    : p.tile_rows == 1 ? Flow::StatsScalar
                                                       : Flow::StatsBlock),
                           p.branches,
                           p.hidden_width,
                           p.splits,
                           p.post_tile_columns,
                           workers};
    return actual_entry(key,
                        fused              ? "mhc_stats_post"
                        : p.tile_rows == 1 ? "mhc_stats_scalar"
                                           : "mhc_stats_block",
                        "mhc/stats.asc", workers, [&] {
                            auto flags = common(p.branches, p.hidden_width, workers);
                            flags.emplace_back("-DMHC_STATS_PLAN=1");
                            if (fused) {
                                flags.emplace_back("-DMHC_STATS_POST=1");
                            } else if (p.tile_rows == 1)
                                flags.emplace_back("-DMHC_STATS_SCALAR=1");
                            field(flags, "TILE_ROWS", fused ? 1 : p.tile_rows);
                            field(flags, "SPLITS", p.splits);
                            field(flags, "COLUMN_TILE", p.post_tile_columns);
                            return flags;
                        });
}
Kernel &state_entry(const StateParams &p, uint32_t workers, Flow flow, bool initial = false,
                    bool normalize = false) {
    const bool single_mean = flow == Flow::Mean && p.rows * p.column_tiles <= workers;
    const Entries::Key key{uint64_t(flow), single_mean,
                           p.branches,
                           p.width,
                           p.column_tile,
                           workers,
                           flow == Flow::Mean ? p.output_columns : 0,
                           initial,
                           normalize};
    const char *tag = flow == Flow::Combine    ? "mhc_combine"
                      : flow == Flow::Collapse ? "mhc_collapse"
                      : flow == Flow::Mean     ? "mhc_mean"
                      : flow == Flow::Post     ? "mhc_post"
                                               : "mhc_post_combine";
    return actual_entry(key, tag, "mhc/state.asc", workers, [&] {
        auto flags = common(p.branches, p.width, workers);
        field(flags, "COLUMN_TILE", p.column_tile);
        field(flags, "OUTPUT_COLUMNS", p.output_columns);
        field(flags, "INITIAL", initial);
        field(flags, "NORMALIZE", normalize);
        if (flow == Flow::Combine)
            flags.emplace_back("-DMHC_COMBINE=1");
        else if (flow == Flow::Collapse)
            flags.emplace_back("-DMHC_COLLAPSE=1");
        else if (flow == Flow::Mean) {
            flags.emplace_back("-DMHC_MEAN=1");
            if (single_mean)
                flags.emplace_back("-DMHC_MEAN_SINGLE_TILE=1");
        } else if (flow == Flow::PostCombine)
            flags.emplace_back("-DMHC_POST_COMBINE=1");
        return flags;
    });
}
} // namespace
extern "C" void dsv41_mhc_project_launch(void *stream, uint32_t blocks, uint8_t *r, uint8_t *fn,
                                         uint8_t *p, uint8_t *s, uint8_t *inputs, uint8_t *products,
                                         mhc::ProjectParams plan, bool vector) {
    const uint64_t row_tiles = (plan.rows + plan.tile_rows - 1) / plan.tile_rows;
    const uint64_t jobs = plan.rows == 0 ? 0 : vector
        ? uint64_t(plan.panel_count) * ((plan.projection_columns + kBlockFloats - 1) / kBlockFloats)
        : plan.splits * std::min<uint64_t>(plan.row_groups, row_tiles);
    if (vector)
        project_entry(plan, blocks, true).launch_persistent(stream, jobs, r, fn, p, s, plan.rows, plan.padded_rows);
    else
        project_entry(plan, blocks, false).launch_persistent(stream, jobs, r, fn, p, s, inputs, products,
            plan.rows, plan.padded_rows);
}
extern "C" void dsv41_mhc_stats_launch(void *stream, uint32_t blocks, uint8_t *p, uint8_t *s,
                                       uint8_t *scale, uint8_t *base, uint8_t *pre, uint8_t *post,
                                       uint8_t *comb, uint8_t *x, uint8_t *r, uint8_t *out,
                                       mhc::StatsParams plan, bool fused) {
    auto &kernel = stats_entry(plan, blocks, fused);
    const uint64_t jobs = fused ? plan.rows * plan.post_column_tiles
                                : (plan.rows + plan.tile_rows - 1) / plan.tile_rows;
    if (fused)
        kernel.launch_persistent(stream, jobs, p, s, scale, base, pre, post, comb, x, r, out,
                      plan.epsilon, plan.rows, plan.padded_rows);
    else
        kernel.launch_persistent(stream, jobs, p, s, scale, base, pre, post, comb,
                      plan.epsilon, plan.rows, plan.padded_rows);
}
extern "C" void dsv41_mhc_combine_launch(void *stream, uint32_t blocks, uint8_t *r, uint8_t *pre,
                                         uint8_t *weight, uint8_t *out,
                                         mhc::StateParams plan, bool initial,
                                         bool normalize) {
    auto &kernel = state_entry(
        plan, blocks, !initial && !normalize ? Flow::Collapse : Flow::Combine, initial, normalize);
    const uint64_t jobs = !initial && !normalize ? plan.rows * plan.column_tiles : plan.rows;
    if (!initial && !normalize)
        kernel.launch_persistent(stream, jobs, r, pre, out, plan.rows);
    else
        kernel.launch_persistent(stream, jobs, r, pre, weight, out, plan.rows);
}
extern "C" void dsv41_mhc_post_launch(void *stream, uint32_t blocks, uint8_t *x, uint8_t *r,
                                      uint8_t *pre, uint8_t *postp, uint8_t *comb, uint8_t *weight,
                                      uint8_t *updated, uint8_t *out, mhc::StateParams plan,
                                      bool fused) {
    auto &kernel = state_entry(plan, blocks, fused ? Flow::PostCombine : Flow::Post);
    const uint64_t jobs = fused ? plan.rows : plan.rows * plan.column_tiles;
    if (fused)
        kernel.launch_persistent(stream, jobs, x, r, pre, postp, comb, weight, updated, out, plan.rows);
    else
        kernel.launch_persistent(stream, jobs, x, r, postp, comb, updated, plan.rows);
}
extern "C" void dsv41_mhc_mean_launch(void *stream, uint32_t blocks, uint8_t *r, uint8_t *out,
                                      mhc::StateParams plan) {
    state_entry(plan, blocks, Flow::Mean).launch_persistent(stream, plan.rows * plan.column_tiles, r, out, plan.column, plan.rows);
}
