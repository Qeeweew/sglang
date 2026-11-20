#pragma once
#include <kernel_operator.h>
#include <common/grouped_metadata.h>

namespace grouped_gemm {
// One monotone task sequence, independently replayed by the AIC and both AIVs.
struct Tile {
    int expert, row, col;
    int valid;
};

// Monotone cursor: each actor visits the compact expert table only once.
// A global cyclic task assignment carries N-tile imbalance across experts.
template <int TileRows, int TileColumns> struct Schedule {
    // Eight-by-eight task swizzle reuses neighboring panels across cores.
    static constexpr int kMinSwizzleRows = 128;
    static constexpr uint32_t kSwizzleTiles = 8;
    __gm__ volatile int32_t *routing;
    int experts, visit = 0, Panels;
    uint32_t Ntiles;
    int prefix = 0, begin = 0, end = 0, expert = 0, tasks = 0, mtiles = 0;
    __aicore__ inline Schedule(__gm__ uint8_t *p, int e, int panels, int n)
        : routing((__gm__ volatile int32_t *)p), experts(e), Panels(panels),
          Ntiles(n / TileColumns) {
        const int active = int(routing[grouped_metadata::kActiveCount]);
        experts = active < e ? active : e;
    }
    // A cursor already on the final nonempty expert knows the exact local
    // extent. This needs no extra route reads or all-expert preprocessing.
    __aicore__ inline int known_steps(int core, int cores) const {
        if (visit < experts)
            return -1;
        return (prefix + tasks - core + cores - 1) / cores * Panels;
    }
    __aicore__ inline bool next(int index, Tile &t) {
        while (index >= prefix + tasks) {
            prefix += tasks;
            if (visit >= experts)
                return false;
            int offset = grouped_metadata::kWords * (1 + visit);
            expert = routing[offset + grouped_metadata::kExpert];
            begin = routing[offset + grouped_metadata::kBegin];
            end = begin + routing[offset + grouped_metadata::kRowCount];
            ++visit;
            // Public contract requires valid sorted, disjoint ranges.
            int count = end > begin ? end - begin : 0;
            mtiles = (count + TileRows - 1) / TileRows;
            tasks = mtiles * Ntiles;
        }
        uint32_t local = uint32_t(index - prefix);
        uint32_t mi = 0, ni = local;
        if (mtiles > 1) {
            mi = local / Ntiles;
            ni = local % Ntiles;
        }
        if constexpr (TileRows >= kMinSwizzleRows) {
            if (mtiles > 1) {
                uint32_t mbase = (mi / kSwizzleTiles) * kSwizzleTiles;
                uint32_t height = mtiles - mbase < kSwizzleTiles ? mtiles - mbase : kSwizzleTiles;
                uint32_t relative = local - mbase * Ntiles;
                uint32_t nbase = (relative / (height * kSwizzleTiles)) * kSwizzleTiles;
                uint32_t width = Ntiles - nbase < kSwizzleTiles ? Ntiles - nbase : kSwizzleTiles;
                uint32_t inner = relative % (height * kSwizzleTiles);
                mi = mbase + inner % height;
                ni = nbase + (inner / height + inner % height) % width;
            }
        }
        t.expert = expert;
        t.row = begin + mi * TileRows;
        t.col = ni * TileColumns;
        t.valid = int(end - t.row < TileRows ? end - t.row : TileRows);
        return true;
    }
};

} // namespace grouped_gemm
