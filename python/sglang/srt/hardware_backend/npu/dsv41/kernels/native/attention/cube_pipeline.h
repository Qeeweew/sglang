#pragma once
#include <common/cube.h>
#include <common/device.h>
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <type_traits>
#include <attention/state.h>

namespace attention {
class CubePipeline {
    const Sources &sources_;
    Workspace workspace_;
    RequestCursor cursor_;
    static constexpr uint32_t kKvSlotBytes = kDotRows * kFeaturePanelColumns * sizeof(BF16);
    int32_t core_, tasks_;
    static constexpr int32_t kCachedProbabilityColumns =
        kCapacity < kWidthCapacity ? kCapacity : kWidthCapacity;
    static constexpr int32_t kQueryProbabilityBytes =
        kTileRows * (kWidth > kCachedProbabilityColumns ? kWidth : kCachedProbabilityColumns) *
        sizeof(BF16);
    static constexpr int32_t kResidentOffset = kQueryProbabilityBytes + 2 * kKvSlotBytes;
    static constexpr int32_t kProbabilityTailOffset =
        kResidentOffset + 2 * kDotRows * kWidth * sizeof(BF16);
    bool previous_full_window_ = false;
    // M -> MTE1 credits for the two L0B operand slots. Each credit is seeded
    // once, consumed before import, reissued after MMAD, then consumed by
    // the final MTE1 drain. QK and PV share this same slot lifetime.
    static constexpr TEventID kB0ReuseEventBase = 4;
    __aicore__ inline __cbuf__ BF16 *l1_pointer(uint32_t bytes) {
        return reinterpret_cast<__cbuf__ BF16 *>(uint64_t(bytes));
    }
    __aicore__ inline __ca__ BF16 *l0a_pointer(uint32_t bytes) {
        return reinterpret_cast<__ca__ BF16 *>(uint64_t(bytes));
    }
    __aicore__ inline __cb__ BF16 *l0b_pointer(uint32_t slot) {
        return reinterpret_cast<__cb__ BF16 *>(uint64_t(slot * kDotRows * kDotRows * sizeof(BF16)));
    }
    __aicore__ inline __cc__ float *l0c_pointer(uint32_t plane) {
        return reinterpret_cast<__cc__ float *>(
            uint64_t(plane * kTileRows * kDotRows * sizeof(float)));
    }
    __aicore__ __attribute__((always_inline)) void copy_kv_rows(
        __cbuf__ BF16 *dst, __gm__ BF16 *source, int32_t destination, int32_t feature,
        int32_t rows) {
        const uint16_t features = min(kFeaturePanelColumns, kWidth - feature);
        asc_copy_gm2l1_nd2nz(dst + destination * (kDmaBytes / sizeof(BF16)),
                             source + feature, 1, uint16_t(rows), features, 0,
                             uint16_t(kWidth), kDotRows, 1, 0);
    }
    __aicore__ __attribute__((always_inline)) void load_paged(__cbuf__ BF16 *dst,
                                                              const RequestMetadata &m,
                                                              int32_t tile, int32_t feature,
                                                              bool ring, int32_t new_rows) {
        int32_t count = min(kDotRows, max(0, m.visible - tile * kDotRows));
        int32_t position = m.first + tile * kDotRows, destination = 0;
        if (ring) {
            position = max(0, m.right - new_rows);
            count = new_rows;
            destination = position % kDotRows;
        }
        if (!count)
            return;
        auto table = gm_pointer<int32_t>(sources_.table);
        auto kv = gm_pointer<BF16>(sources_.kv);
        int32_t logical_page = kPageShift >= 0 ? position >> kPageShift : position / kPageRows;
        int32_t page_offset = position - logical_page * kPageRows;
        for (;;) {
            const int32_t page =
                table[int64_t(m.batch) * kPageColumns + min(logical_page, kPageColumns - 1)];
            const int32_t chunk = min(count, min(kPageRows - page_offset, kDotRows - destination));
            copy_kv_rows(dst, kv + (int64_t(page) * kPageRows + page_offset) * kWidth,
                         destination, feature, chunk);
            // The final transfer has no successor address to prepare. Across
            // later chunks, advance the bounded page cursor instead of
            // repeating the initial shift/division for every DMA.
            if (count == chunk)
                break;
            count -= chunk;
            page_offset += chunk;
            if (page_offset == kPageRows) {
                ++logical_page;
                page_offset = 0;
            }
            destination = (destination + chunk) % kDotRows;
        }
    }
    __aicore__ __attribute__((always_inline)) void load_verify_window(
        __cbuf__ BF16 *dst, const RequestMetadata &m, int32_t feature) {
        if (!m.visible)
            return;
        const int32_t query_begin = gm_pointer<int32_t>(sources_.cu)[m.batch];
        const int32_t prefix = m.right - m.offset - 1;
        const int64_t request = gm_pointer<int64_t>(sources_.verify_requests)[m.batch];
        int32_t position = m.first;
        // Preserve the paged kernel's rotated L1 order for single-request
        // sparse attention. The source can wrap independently in the ring.
        while (position < m.right) {
            const int32_t destination = kWindowRing ? position % kDotRows : position - m.first;
            const bool pending = position >= prefix;
            const int32_t offset = position % kVerifyRingCapacity;
            const int32_t available = pending ? m.right - position
                                               : min(prefix - position, kVerifyRingCapacity - offset);
            const int32_t rows = min(available, kDotRows - destination);
            auto source = pending
                              ? gm_pointer<BF16>(sources_.verify_pending) +
                                    int64_t(query_begin + position - prefix) * kWidth
                              : gm_pointer<BF16>(sources_.kv) +
                                    (request * kVerifyRingCapacity + offset) * kWidth;
            copy_kv_rows(dst, source, destination, feature, rows);
            position += rows;
        }
    }
    __aicore__ __attribute__((always_inline)) __cbuf__ BF16 *
    load_panel(int32_t task, const RequestMetadata &m, int32_t tile, int32_t feature, int32_t slot,
               int32_t new_rows) {
        const bool resident = tile < 2;
        auto dst = l1_pointer(resident ? kResidentOffset + tile * kDotRows * kWidth * sizeof(BF16) +
                                             feature * kDotRows * sizeof(BF16)
                                       : kQueryProbabilityBytes + slot * kKvSlotBytes);
        if constexpr (kSparse) {
            if (tile > 0) {
                asc_copy_gm2l1_nd2nz(
                    dst, (workspace_.gathered(task) + ((tile - 1) * kDotRows * kWidth + feature)),
                    1, kDotRows, uint16_t(min(kFeaturePanelColumns, kWidth - feature)), 0,
                    uint16_t(kWidth), kDotRows, 1, 0);
            } else if constexpr (kVerifyRingCapacity) {
                load_verify_window(dst, m, feature);
            } else {
                load_paged(dst, m, tile, feature, kWindowRing, new_rows);
            }
        } else if constexpr (kVerifyRingCapacity) {
            load_verify_window(dst, m, feature);
        } else {
            load_paged(dst, m, tile, feature, false, new_rows);
        }
        return dst;
    }
    __aicore__ inline void load_left(__ca__ BF16 *dst, __cbuf__ BF16 *src, int32_t depth) {
        if constexpr (kTileRows == kMatrixFractalRows) {
            asc_copy_l12l0a(reinterpret_cast<__ca__ half *>(dst),
                            reinterpret_cast<__cbuf__ half *>(src), 0,
                            uint8_t(kTileRows * depth / kMatrixFractalElements), 1, 0);
        } else {
            asc_copy_l12l0a(reinterpret_cast<__ca__ half *>(dst),
                            reinterpret_cast<__cbuf__ half *>(src), depth, kTileRows, 0, 0, 1, 1, 1,
                            1, 1, 1, false, false, false, false, depth);
        }
    }
    template <class Output>
    __aicore__ __attribute__((always_inline)) void
    store_matrix(__gm__ Output *dst, __cc__ float *src, uint32_t columns, uint32_t stride) {
        // SDK C++ FixpipeL0C2GMImpl unconditionally reprograms ND_PARA
        // and inserts PIPE_FIX on every store. Our row-major ND parameters
        // are invariant; publish them once in run and keep only the public
        // C transfer primitive here. The owning stages already order M/FIX
        // and retire every L0C plane before reuse.
        constexpr uint64_t quant =
            std::is_same<Output, BF16>::value ? uint64_t(QuantMode_t::F322BF16) : 0;
        asc_copy_l0c2gm(dst, src, uint16_t(columns), uint16_t(kHeads), stride, uint16_t(kTileRows),
                        0, quant, 0, false, true);
    }
    __aicore__ __attribute__((always_inline)) void query_key(int32_t task) {
        const int32_t token = core_ + task * kWorkerBudget;
        const auto m = cursor_.locate(sources_, token);
        int32_t new_window_rows = m.visible;
        if constexpr (kSparse) {
            if (kWindowRing) {
                const bool full = m.visible == min(int32_t(kWindowRows), m.right);
                if (task && full && previous_full_window_)
                    new_window_rows = min(m.visible, int32_t(kWorkerBudget));
                previous_full_window_ = full;
            }
        }
        // The first task has no earlier L1 reader. Later tasks retire the
        // previous PV import before the Q/P arena is overwritten.
        if (task) {
            native_device::pipe_fence<PIPE_MTE1, PIPE_MTE2, kImmediateEvent>();
        }
        asc_copy_gm2l1_nd2nz(
            l1_pointer(0), (gm_pointer<BF16>(sources_.q) + (int64_t(token) * kHeads * kWidth)), 1,
            uint16_t(kHeads), uint16_t(kWidth), 0, uint16_t(kWidth), uint16_t(kTileRows), 1, 0);
        native_device::pipe_fence<PIPE_MTE2, PIPE_MTE1, kImmediateEvent>();
        const int32_t query_parts = (kWidth + kDotRows - 1) / kDotRows;
#pragma unroll
        for (int32_t part = 0; part < kWidthCapacity / kDotRows; ++part) {
            if (part >= query_parts)
                continue;
            if (!part && task) {
                native_device::pipe_fence<PIPE_M, PIPE_MTE1, kQueryReuseEvent>();
            }
            load_left(l0a_pointer(part * kTileRows * kDotRows * sizeof(BF16)),
                      l1_pointer(part * kTileRows * kDotRows * sizeof(BF16)),
                      min(kDotRows, kWidth - part * kDotRows));
        }
        // The first L0B import's MTE1->M event also publishes all preceding
        // Q imports, so a separate Q-only event would duplicate that lifetime.
        for (int32_t tile = 0; tile < kCapacity / kDotRows; ++tile) {
            if constexpr (kSparse)
                if (tile == 1)
                    CrossCoreWaitFlag<2, PIPE_MTE2>(GatherReady);
            __cbuf__ BF16 *planes[2];
            const int32_t panels = (kWidth + kFeaturePanelColumns - 1) / kFeaturePanelColumns;
#pragma unroll
            for (int32_t panel = 0; panel < kWidthCapacity / kFeaturePanelColumns; ++panel) {
                if (panel >= panels)
                    continue;
                planes[panel] =
                    load_panel(task, m, tile, panel * kFeaturePanelColumns, panel, new_window_rows);
                set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(panel + 2));
            }
#pragma unroll
            for (int32_t panel = 0; panel < kWidthCapacity / kFeaturePanelColumns; ++panel) {
                if (panel >= panels)
                    continue;
                wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(panel + 2));
                const int32_t parts = min(2, (kWidth + kDotRows - 1) / kDotRows - panel * 2);
#pragma unroll
                for (int32_t part = 0; part < kFeaturePanelColumns / kDotRows; ++part) {
                    if (part >= parts)
                        continue;
                    wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase + part));
                    const int32_t depth = min(kDotRows, kWidth - (panel * 2 + part) * kDotRows);
                    asc_copy_l12l0b(l0b_pointer(part),
                                    (planes[panel] + (part * kDotRows * kDotRows)), 0,
                                    uint8_t(kDotRows * depth / kMatrixFractalElements), 1, 0);
                    native_device::pipe_fence<PIPE_MTE1, PIPE_M, kImmediateEvent>();
                    if (!panel && !part && tile >= 2)
                        wait_flag(PIPE_FIX, PIPE_M, ::event_t((tile % 2) + 6));
                    native_device::mmad(
                        l0c_pointer(tile % 2),
                        l0a_pointer((panel * 2 + part) * kTileRows * kDotRows * sizeof(BF16)),
                        l0b_pointer(part), kTileRows, depth, kDotRows, 0, !panel && !part);
                    set_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase + part));
                }
            }
            native_device::pipe_fence<PIPE_M, PIPE_FIX, kImmediateEvent>();
            store_matrix((workspace_.scores(task) + (tile * kDotRows)), l0c_pointer(tile % 2),
                         kDotRows, kCapacity);
            set_flag(PIPE_FIX, PIPE_M, ::event_t((tile % 2) + 6));
            // Resident tiles own disjoint L1 regions. Only streamed QK
            // panels reuse the same two feature slots at the next tile.
            // At the final tile, ScoreReady -> P publication -> ProbabilityReady
            // completes the readers before PV imports P into the Q/P arena.
            if (tile >= 2 && tile + 1 < kCapacity / kDotRows) {
                native_device::pipe_fence<PIPE_MTE1, PIPE_MTE2, kImmediateEvent>();
            }
        }
        CrossCoreSetFlag<2, PIPE_FIX>(ScoreReady);
    }
    __aicore__ __attribute__((always_inline)) void probability_value(int32_t task) {
        CrossCoreWaitFlag<2, PIPE_MTE2>(ProbabilityReady);
        const int32_t token = core_ + task * kWorkerBudget, tiles = kCapacity / kDotRows;
        RequestMetadata m{};
        if constexpr (kDraft)
            if (kCapacity > 2 * kDotRows)
                m = cursor_.locate(sources_, token);
        if constexpr (!kSparse && !kDraft)
            if (kCapacity > 2 * kDotRows)
                m = cursor_.locate(sources_, token);
        const int32_t cached = min(int32_t(kCapacity), kWidthCapacity);
        for (int32_t tile = max(0, tiles - 2); tile < tiles; ++tile)
            wait_flag(PIPE_FIX, PIPE_M, ::event_t((tile % 2) + 6));
        asc_copy_gm2l1_nd2nz(l1_pointer(0), workspace_.probabilities(task), 1, uint16_t(kHeads),
                             uint16_t(cached), 0, uint16_t(kCapacity), uint16_t(kTileRows), 1, 0);
        native_device::pipe_fence<PIPE_MTE2, PIPE_MTE1, kImmediateEvent>();
        for (int32_t column = 0; column < kWidth; column += kFeaturePanelColumns) {
            const int32_t parts = min(2, (kWidth - column + kDotRows - 1) / kDotRows);
            for (int32_t step = 0; step <= tiles; ++step) {
                if (step < tiles) {
                    const int32_t slot = step % 2;
                    if (step >= 4)
                        wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot + 1));
                    if (step >= cached / kDotRows) {
                        asc_copy_gm2l1_nd2nz(l1_pointer(kProbabilityTailOffset +
                                                        slot * kTileRows * kDotRows * sizeof(BF16)),
                                             (workspace_.probabilities(task) + (step * kDotRows)),
                                             1, uint16_t(kHeads), kDotRows, 0, uint16_t(kCapacity),
                                             uint16_t(kTileRows), 1, 0);
                    }
                    if (step >= 2) {
                        load_panel(task, m, step, column, slot, m.visible);
                        set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot + 1));
                    }
                }
                if (step > 0) {
                    const int32_t tile = step - 1, slot = tile % 2;
                    if (tile >= 2)
                        wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot + 1));
                    wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase));
                    auto p = l1_pointer(tile < cached / kDotRows
                                            ? tile * kTileRows * kDotRows * sizeof(BF16)
                                            : kProbabilityTailOffset +
                                                  slot * kTileRows * kDotRows * sizeof(BF16));
                    auto a = l0a_pointer(((column / kFeaturePanelColumns * tiles + tile) % 2) *
                                         kTileRows * kDotRows * sizeof(BF16));
                    load_left(a, p, kDotRows);
                    auto values = l1_pointer(
                        tile < 2 ? kResidentOffset + tile * kDotRows * kWidth * sizeof(BF16) +
                                       column * kDotRows * sizeof(BF16)
                                 : kQueryProbabilityBytes + slot * kKvSlotBytes);
#pragma unroll
                    for (int32_t part = 0; part < kFeaturePanelColumns / kDotRows; ++part) {
                        if (part >= parts)
                            continue;
                        if (part)
                            wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase + 1));
                        const int32_t features = min(kDotRows, kWidth - column - part * kDotRows);
                        // Preserve half bit views and FMATRIX_B selection; no fractal transpose.
                        asc_copy_l12l0b(
                            reinterpret_cast<__cb__ half *>(l0b_pointer(part)),
                            reinterpret_cast<__cbuf__ half *>(values + part * kDotRows * kDotRows),
                            features, kDotRows, 0, 0, 1, 1, 1, 1, 1, 1, false, false, false, true,
                            features);
                        if (part + 1 == parts && tile >= 2)
                            set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot + 1));
                        native_device::pipe_fence<PIPE_MTE1, PIPE_M, kImmediateEvent>();
                        native_device::mmad(l0c_pointer(part), a, l0b_pointer(part), kTileRows,
                                            kDotRows, features, 0, tile == 0);
                        set_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase + part));
                    }
                }
            }
            for (int32_t slot = max(2, tiles - 2); slot < tiles; ++slot)
                wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t((slot % 2) + 1));
            native_device::pipe_fence<PIPE_M, PIPE_FIX, kImmediateEvent>();
            if constexpr (kSparse)
                store_matrix((workspace_.products(task) + (column)), l0c_pointer(0),
                             min(kFeaturePanelColumns, kWidth - column), kWidth);
            else
                store_matrix(
                    (gm_pointer<BF16>(sources_.out) + (int64_t(token) * kHeads * kWidth + column)),
                    l0c_pointer(0), min(kFeaturePanelColumns, kWidth - column), kWidth);
            native_device::pipe_fence<PIPE_FIX, PIPE_M, kImmediateEvent>();
        }
        if constexpr (kSparse)
            CrossCoreSetFlag<2, PIPE_FIX>(ProductReady);
    }

  public:
    __aicore__
        __attribute__((always_inline)) explicit CubePipeline(const Sources &args, int32_t actor)
        : sources_(args), workspace_(args, actor), core_(actor) {
        tasks_ = sources_.tokens <= kWorkerBudget
                     ? int32_t(core_ < sources_.tokens)
                     : (max(0, sources_.tokens - core_) + kWorkerBudget - 1) / kWorkerBudget;
    }
    __aicore__ __attribute__((always_inline)) void run() {
        asc_set_l0c_copy_params(1, 0, 0);
        native_device::pipe_fence<PIPE_S, PIPE_FIX, kImmediateEvent>();
        // Entry InitSocState owns zero PADDING/L1_3D_SIZE on arch2201.
        // No stage changes them; FMATRIX and L3D_RPT still need setup here.
        set_l3d_rpt(uint64_t(1) << 16);
        // CANN encodes W in bits[15:0], H in bits[31:16]. Each NZ
        // operand is one spatial row, matching rows | (1 << 16).
        set_fmatrix(uint64_t(kTileRows) | (uint64_t(1) << 16));
        set_fmatrix_b(uint64_t(kDotRows) | (uint64_t(1) << 16));
        native_device::pipe_fence<PIPE_S, PIPE_MTE1, kImmediateEvent>();
        // Only the direct window needs padding initialization. Sparse selected
        // panels are fully replaced by gather; window/draft need at most two.
        const int32_t resident_rows = kSparse ? kDotRows : min(int32_t(kCapacity), 2 * kDotRows);
        {
            asc_fill_value_config fill{};
            fill.repeat = 1;
            fill.blk_num = uint16_t(resident_rows * kWidth * sizeof(BF16) / kDmaBytes);
            fill.dst_gap = 0;
            asc_fill_l1(reinterpret_cast<__cbuf__ uint16_t *>(l1_pointer(kResidentOffset)),
                        uint32_t(0), fill);
        };
        if (kTileRows > kHeads) {
            // NZ is [depth/16, tile_rows, 16]. Real heads are overwritten
            // by the Q/P DMAs; clear only padded head lanes in each fractal.
            const int32_t depth = max(kWidth, min(int32_t(kCapacity), kWidthCapacity));
            {
                asc_fill_value_config fill{};
                fill.repeat = uint16_t(depth / kMatrixFractalRows);
                fill.blk_num = uint16_t(kTileRows - kHeads);
                fill.dst_gap = uint16_t(kHeads);
                asc_fill_l1(reinterpret_cast<__cbuf__ uint16_t *>(l1_pointer(kHeads * kDmaBytes)),
                            uint32_t(0), fill);
            };
        }
        if constexpr (!kSparse)
            if (kCapacity > 2 * kDotRows) {
                asc_fill_value_config fill{};
                fill.repeat = 1;
                fill.blk_num = uint16_t(2 * kKvSlotBytes / kDmaBytes);
                fill.dst_gap = 0;
                asc_fill_l1(
                    reinterpret_cast<__cbuf__ uint16_t *>(l1_pointer(kQueryProbabilityBytes)),
                    uint32_t(0), fill);
            };
        // First query_key publishes both these initial fills and its Q DMA
        // through one MTE2->MTE1 event before any operand import.
        set_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase));
        set_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase + 1));
        // A single task has no ring generation or next-query lifetime.
        // Share the same stages, with a constant first/final task index.
        if (!kSparse && tasks_ == 1) {
            query_key(0);
            probability_value(0);
        } else
            for (int32_t step = 0; step <= tasks_; ++step) {
                if (step > 0)
                    probability_value(step - 1);
                if (step < tasks_)
                    query_key(step);
            }
        wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase));
        wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kB0ReuseEventBase + 1));
        // The last streamed PV tile also consumes free-slot credits on
        // MTE2. Retire those Wait instructions themselves, even after their
        // MTE1 producers and the final matrix export have completed.
        if (kCapacity > 2 * kDotRows) {
            native_device::pipe_fence<PIPE_MTE2, PIPE_S, kImmediateEvent>();
        }
        // Consume the two L0B credits on MTE1 before returning. The final PV
        // export already retired all operand imports through MTE1->M and
        // M->FIX; its FIX->M wait makes the M drain cover that export too.
        native_device::pipe_fence<PIPE_MTE1, PIPE_S, kImmediateEvent>();
        native_device::pipe_fence<PIPE_M, PIPE_S, kImmediateEvent>();
    }
};
} // namespace attention
