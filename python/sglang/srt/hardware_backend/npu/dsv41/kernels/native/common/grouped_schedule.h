#pragma once
#include <kernel_operator.h>
#include <common/grouped_metadata.h>

namespace grouped_gemm {
// One monotone task sequence, independently replayed by the AIC and both AIVs.
struct Tile {
    int expert, row, col;
    int valid;
};

// Monotone cursor: each actor visits the compact expert table once.
//
// In prefill, when Ntiles is a multiple of the worker count, cyclic row-major
// places every worker on the same cold activation row tile simultaneously,
// while 8x8 swizzle resonates modulo workers (skewing full vs. tail rows and
// breaking L1 activation reuse when workers % mtiles != 0). Grouping contiguous
// N-tiles per worker with wave rotation avoids both effects while keeping
// workers inside 1-2 experts for L2 weight reuse.
template <int TileRows, int TileColumns> struct Schedule {
    static constexpr int kMinSwizzleRows = 128;
    static constexpr int kMinChunkMeanRows = 48;
    static constexpr uint32_t kSwizzleTiles = 8;
    __gm__ volatile int32_t *routing;
    int experts, visit = 0, Panels;
    uint32_t Ntiles;
    int prefix = 0, begin = 0, end = 0, expert = 0, tasks = 0, mtiles = 0;

    // Prefill N-chunk state (inactive when chunk_waves_ == 0).
    int core_id_ = 0, num_cores_ = 1;
    int chunk_size_ = 0, chunk_waves_ = 0, rem_base_ = 0, total_tasks_ = 0;
    int wave_ = 0, inner_ = 0, slot_ = 0;
    int last_index_ = -1;
    Tile last_tile_{};

    __aicore__ inline static int select_chunk(uint32_t ntiles, int cores) {
        if (cores <= 0 || ntiles == 0 || ntiles % uint32_t(cores) != 0)
            return 0;
        constexpr int candidates[] = {10, 8, 5, 4};
        for (int g : candidates)
            if (ntiles % uint32_t(g) == 0 && int(ntiles) >= g)
                return g;
        return 0;
    }

    __aicore__ inline Schedule(__gm__ uint8_t *p, int e, int panels, int n, int core, int cores)
        : routing((__gm__ volatile int32_t *)p), experts(e), Panels(panels),
          Ntiles(uint32_t(n / TileColumns)), core_id_(core), num_cores_(cores), slot_(core) {
        const int active = int(routing[grouped_metadata::kActiveCount]);
        experts = active < e ? active : e;
        if constexpr (TileRows >= kMinSwizzleRows) {
            const int chunk = select_chunk(Ntiles, cores);
            const int total_rows = int(routing[grouped_metadata::kTotalRows]);
            const int wave_span = cores * chunk;
            const int64_t max_tasks =
                (int64_t(total_rows) / TileRows + experts) * int64_t(Ntiles);
            if (chunk > 0 && experts > 0 && max_tasks >= wave_span &&
                int64_t(total_rows) >= int64_t(experts) * kMinChunkMeanRows) {
                int tot = 0;
                for (int v = 0; v < experts; ++v) {
                    const int count =
                        int(routing[grouped_metadata::kWords * (1 + v) + grouped_metadata::kRowCount]);
                    if (count > 0)
                        tot += ((count + TileRows - 1) / TileRows) * int(Ntiles);
                }
                const int waves = tot / wave_span;
                if (waves > 0) {
                    chunk_size_ = chunk;
                    chunk_waves_ = waves;
                    rem_base_ = waves * wave_span;
                    total_tasks_ = tot;
                }
            }
        }
    }

    // A cursor already on the final nonempty expert knows the exact local
    // extent. This needs no extra route reads or all-expert preprocessing.
    __aicore__ inline int known_steps(int core, int cores) const {
        if (chunk_waves_ > 0 || visit < experts)
            return -1;
        return (prefix + tasks - core + cores - 1) / cores * Panels;
    }

    __aicore__ inline bool map_linear(int index, Tile &t) {
        while (index >= prefix + tasks) {
            prefix += tasks;
            if (visit >= experts)
                return false;
            const int offset = grouped_metadata::kWords * (1 + visit);
            expert = routing[offset + grouped_metadata::kExpert];
            begin = routing[offset + grouped_metadata::kBegin];
            end = begin + routing[offset + grouped_metadata::kRowCount];
            ++visit;
            // Public contract requires valid sorted, disjoint ranges.
            const int count = end > begin ? end - begin : 0;
            mtiles = (count + TileRows - 1) / TileRows;
            tasks = mtiles * int(Ntiles);
        }
        const uint32_t local = uint32_t(index - prefix);
        uint32_t mi = 0, ni = local;
        if (mtiles > 1) {
            mi = local / Ntiles;
            ni = local % Ntiles;
        }
        if constexpr (TileRows >= kMinSwizzleRows) {
            if (chunk_waves_ == 0 && mtiles > 1) {
                const uint32_t mbase = (mi / kSwizzleTiles) * kSwizzleTiles;
                const uint32_t height =
                    mtiles - mbase < kSwizzleTiles ? mtiles - mbase : kSwizzleTiles;
                const uint32_t relative = local - mbase * Ntiles;
                const uint32_t nbase = (relative / (height * kSwizzleTiles)) * kSwizzleTiles;
                const uint32_t width =
                    Ntiles - nbase < kSwizzleTiles ? Ntiles - nbase : kSwizzleTiles;
                const uint32_t inner = relative % (height * kSwizzleTiles);
                mi = mbase + inner % height;
                ni = nbase + (inner / height + inner % height) % width;
            }
        }
        t.expert = expert;
        t.row = begin + int(mi) * TileRows;
        t.col = int(ni) * TileColumns;
        t.valid = end - t.row < TileRows ? end - t.row : TileRows;
        return true;
    }

    __aicore__ inline bool next(int index, Tile &t) {
        if constexpr (TileRows >= kMinSwizzleRows) {
            if (chunk_waves_ > 0) {
                if (index == last_index_) {
                    t = last_tile_;
                    return true;
                }
                int linear;
                if (wave_ < chunk_waves_) {
                    linear = (wave_ * num_cores_ + slot_) * chunk_size_ + inner_;
                    if (++inner_ == chunk_size_) {
                        inner_ = 0;
                        ++wave_;
                        if (++slot_ == num_cores_)
                            slot_ = 0;
                    }
                } else {
                    linear = rem_base_ + inner_ * num_cores_ + core_id_;
                    ++inner_;
                    if (linear >= total_tasks_)
                        return false;
                }
                if (!map_linear(linear, t))
                    return false;
                last_index_ = index;
                last_tile_ = t;
                return true;
            }
        }
        return map_linear(index, t);
    }
};

} // namespace grouped_gemm
