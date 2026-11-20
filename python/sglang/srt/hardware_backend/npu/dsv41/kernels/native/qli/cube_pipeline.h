#pragma once
#include <common/cube.h>
#include <common/device.h>
#include <qli/schedule.h>

namespace qli {
class CubePipeline {
    const Sources &args_;
    Workspace workspace_;
    // Distinct local event pairs own the two prefetch spans. Event 0 remains
    // the short-lived stage dependency used by the other imports.
    static constexpr uint32_t kKeyEvent = 2;
    static constexpr uint32_t kReluEvent = 2;
    static constexpr uint32_t kQkEvent = 4;
    static constexpr uint32_t kHeadEvent = 5;
    static_assert(kKeySlots == 2 && kReluSlots == 2);

    __aicore__ void load_query(const Task &task) {
        // Prior task's Q/W imports must retire before their L1 spans change.
        native_device::pipe_fence<PIPE_MTE1, PIPE_MTE2, kImmediateEvent>();
        if (task.queries < kQueries) {
            asc_fill_value_config fill{};
            fill.repeat = 1;
            fill.blk_num = uint16_t(kMatrixRows * kDepth / 32);
            fill.dst_gap = 0;
            asc_fill_l1(l1_pointer<uint16_t>(L1Layout::query), uint32_t(0), fill);
        };
        asc_copy_gm2l1_nd2nz(l1_pointer<int8_t>(L1Layout::query),
                             (gm_pointer<int8_t>(args_.q) + (uint64_t(task.row) * kHeads * kDepth)),
                             1, uint16_t(task.queries * kHeads), uint16_t(kDepth), 0,
                             uint16_t(kDepth), uint16_t(kMatrixRows), 1, 0);
        CrossCoreWaitFlag<2, PIPE_MTE2>(WeightReady);
        asc_copy_gm2l1_nd2nz(l1_pointer<half>(L1Layout::weights), workspace_.weights(), 1,
                             kCubeRows, uint16_t(kMatrixRows), 0, uint16_t(kMatrixRows), kCubeRows,
                             1, 0);
        // GM weights are task-local input after this DMA. The vector actors
        // may now prepare the next task in the same fixed-address GM span.
        CrossCoreSetFlag<2, PIPE_MTE2>(WeightCredit);
        native_device::pipe_fence<PIPE_MTE2, PIPE_MTE1, kImmediateEvent>();
        if constexpr (kCachedLeftOperands) {
            native_device::pipe_fence<PIPE_M, PIPE_MTE1, kImmediateEvent>();
            load_query_operand();
            asc_copy_l12l0a(l0a_pointer<half>(kWeightOperandOffset),
                            l1_pointer<half>(L1Layout::weights), 0, uint8_t(kMatrixRows / 16), 1,
                            0);
        }
    }
    __aicore__ void load_key(const Task &task, uint32_t begin, uint32_t count, uint32_t slot,
                             bool reused) {
        if (reused)
            wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(kKeyEvent + slot));
        auto pages = gm_pointer<int32_t>(args_.pages);
        auto keys = gm_pointer<int8_t>(args_.k);
        uint32_t destination = 0;
        while (destination < count) {
            const uint32_t position = begin + destination;
            const uint32_t page_column = position / kPageRows;
            const uint32_t page_row = position - page_column * kPageRows;
            const uint32_t page = pages[uint64_t(task.batch) * kPageColumns + page_column];
            const uint32_t rows = min(count - destination, kPageRows - page_row);
            asc_copy_gm2l1_nd2nz(
                (l1_pointer<int8_t>(L1Layout::key) +
                 (slot * kKeyRows * kDepth + destination * kInt8DepthAlignment)),
                (keys + (uint64_t(page) * kKeyPageStride + uint64_t(page_row) * kDepth)), 1,
                uint16_t(rows), uint16_t(kDepth), 0, uint16_t(kDepth),
                uint16_t((count + kCubeRows - 1) / kCubeRows * kCubeRows), 1, 0);
            destination += rows;
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(kKeyEvent + slot));
    }
    __aicore__ void load_query_operand() {
        asc_copy_l12l0a(l0a_pointer<int8_t>(), l1_pointer<int8_t>(L1Layout::query), kDepth,
                        kMatrixRows, 0, 0, 1, 1, 1, 1, 1, 1, false, false, false, false, kDepth);
    }
    template <bool Full>
    __aicore__ void query_key(uint32_t tail_columns, uint32_t slot, bool reused, bool relu_reused) {
        const uint32_t columns = Full ? kKeyRows : tail_columns;
        wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(kKeyEvent + slot));
        if constexpr (kPipelinedRightOperands) {
            if (reused)
                wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kQkEvent));
        } else {
            // The preceding head MMAD must release the shared L0B operand.
            native_device::pipe_fence<PIPE_M, PIPE_MTE1, kImmediateEvent>();
        }
        if constexpr (!kCachedLeftOperands)
            load_query_operand();
        asc_copy_l12l0b(l0b_pointer<int8_t>(),
                        (l1_pointer<int8_t>(L1Layout::key) + (slot * kKeyRows * kDepth)), 0,
                        uint8_t(((columns + 15) / 16) * (kDepth / 32)), 1, 0);
        set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(kKeyEvent + slot));
        native_device::pipe_fence<PIPE_MTE1, PIPE_M, kImmediateEvent>();
        if constexpr (kPipelinedRightOperands)
            if (reused)
                wait_flag(PIPE_FIX, PIPE_M, ::event_t(kQkEvent));
        native_device::mmad<native_device::MmadColumns::Tail>(
            l0c_pointer<int32_t>(), l0a_pointer<int8_t>(), l0b_pointer<int8_t>(), kMatrixRows,
            kDepth, columns, uint8_t(0), true);
        if constexpr (kPipelinedRightOperands)
            set_flag(PIPE_M, PIPE_MTE1, ::event_t(kQkEvent));
        native_device::pipe_fence<PIPE_M, PIPE_FIX, kImmediateEvent>();
        set_quant_pre(0x3a800000); // exact FP32 1/1024 before FP16 rounding
        if constexpr (kPipelinedRightOperands)
            if (relu_reused)
                wait_flag(PIPE_MTE1, PIPE_FIX, ::event_t(kReluEvent + slot));
        asc_copy_l0c2l1((l1_pointer<half>(L1Layout::relu) +
                         (kPipelinedRightOperands ? slot * kMatrixRows * kKeyRows : 0)),
                        l0c_pointer<int32_t>(), uint16_t((columns + 15) / 16 * 16), kMatrixRows,
                        kMatrixRows, kMatrixRows, 0, uint64_t(QuantMode_t::DEQF16), 1, false,
                        false);
        if constexpr (kPipelinedRightOperands) {
            set_flag(PIPE_FIX, PIPE_MTE1, ::event_t(kReluEvent + slot));
            set_flag(PIPE_FIX, PIPE_M, ::event_t(kQkEvent));
        } else {
            native_device::pipe_fence<PIPE_FIX, PIPE_MTE1, kImmediateEvent>();
        }
    }
    __aicore__ void weight_heads(const Task &task, uint32_t columns, uint32_t slot,
                                 uint32_t destination, uint32_t relu_slot, bool reused) {
        if constexpr (kPipelinedRightOperands) {
            if (reused)
                wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kHeadEvent));
            wait_flag(PIPE_FIX, PIPE_MTE1, ::event_t(kReluEvent + relu_slot));
        } else {
            native_device::pipe_fence<PIPE_M, PIPE_MTE1, kImmediateEvent>();
        }
        // One diagonal weight matrix handles all query rows in one MMAD.
        // Its zero off-diagonal heads preserve each independent head sum.
        if constexpr (!kCachedLeftOperands)
            asc_copy_l12l0a(l0a_pointer<half>(), l1_pointer<half>(L1Layout::weights), 0,
                            uint8_t(kMatrixRows / 16), 1, 0);
        const uint16_t padded_columns = (columns + kCubeRows - 1) / kCubeRows * kCubeRows;
        asc_copy_l12l0b(l0b_pointer<half>(kHeadOperandOffset),
                        l1_pointer<half>(L1Layout::relu) +
                            (kPipelinedRightOperands ? relu_slot * kMatrixRows * kKeyRows : 0),
                        padded_columns, kMatrixRows, 0, 0, 1, 1, 1, 1, 1, 1, false, false, true,
                        true, padded_columns);
        if constexpr (kPipelinedRightOperands)
            set_flag(PIPE_MTE1, PIPE_FIX, ::event_t(kReluEvent + relu_slot));
        native_device::pipe_fence<PIPE_MTE1, PIPE_M, kImmediateEvent>();
        auto result = l0c_pointer<float>(arch2201::kL0CBytes / 2);
        if constexpr (kPipelinedRightOperands)
            if (reused)
                wait_flag(PIPE_FIX, PIPE_M, ::event_t(kHeadEvent));
        native_device::mmad<native_device::MmadColumns::Tail>(
            result, l0a_pointer<half>(kWeightOperandOffset), l0b_pointer<half>(kHeadOperandOffset),
            kCubeRows, kMatrixRows, columns, uint8_t(0), true);
        if constexpr (kPipelinedRightOperands)
            set_flag(PIPE_M, PIPE_MTE1, ::event_t(kHeadEvent));
        native_device::pipe_fence<PIPE_M, PIPE_FIX, kImmediateEvent>();
        asc_copy_l0c2gm((workspace_.scores(slot) + (destination)), result, uint16_t(columns),
                        uint16_t(task.queries), kKeyTileCapacity, kCubeRows, 0,
                        uint64_t(QuantMode_t::NoQuant), 0, false, true);
        if constexpr (kPipelinedRightOperands)
            set_flag(PIPE_FIX, PIPE_M, ::event_t(kHeadEvent));
        // In the shared-operand path, the next QK FIX -> MTE1 -> M chain
        // retires this export before the next head overwrites its L0C half.
    }

  public:
    __aicore__ CubePipeline(const Sources &args, uint32_t core)
        : args_(args), workspace_{args, core} {}
    __aicore__ void run() {

        // InitSocState owns zero padding/boundary, not matrix/repeat/FIX state.
        set_fmatrix(uint64_t(kMatrixRows) | (uint64_t(1) << 16));
        set_fmatrix_b(uint64_t(kMatrixRows) | (uint64_t(1) << 16));
        set_l3d_rpt(uint64_t(1) << 16);
        asc_set_l0c_copy_params(1, 0, 0);
        native_device::pipe_fence<PIPE_S, PIPE_MTE1, kImmediateEvent>();
        native_device::pipe_fence<PIPE_S, PIPE_FIX, kImmediateEvent>();
        TaskCursor cursor(args_);
        Task task{};
        uint64_t published = 0;
        for (uint64_t ordinal = workspace_.core; cursor.locate(ordinal, task);
             ordinal += kScoreWorkers) {
            if (task.key_begin >= task.key_end)
                continue;
            load_query(task);
            for (uint32_t begin = task.key_begin; begin < task.key_end;
                 begin += kKeyTileCapacity, ++published) {
                const uint32_t slot = published % kScoreSlots;
                if (published >= kScoreSlots)
                    CrossCoreWaitFlag<2, PIPE_MTE2>(slot ? ScoreCredit1 : ScoreCredit0);
                const uint32_t end = min(task.key_end, begin + kKeyTileCapacity);
                load_key(task, begin, min(uint32_t(kKeyRows), end - begin), 0, false);
                uint32_t tile = 0;
                for (uint32_t key = begin; key < end; key += kKeyRows) {
                    const uint32_t columns = min(uint32_t(kKeyRows), end - key);
                    const uint32_t key_slot = tile % kKeySlots;
                    const uint32_t next = key + kKeyRows;
                    if (next < end)
                        load_key(task, next, min(uint32_t(kKeyRows), end - next), 1 - key_slot,
                                 tile != 0);
                    if (key + kKeyRows <= end)
                        query_key<true>(kKeyRows, key_slot, tile != 0, tile >= kReluSlots);
                    else
                        query_key<false>(columns, key_slot, tile != 0, tile >= kReluSlots);
                    if constexpr (kPipelinedRightOperands) {
                        if (tile)
                            weight_heads(task, kKeyRows, slot, key - kKeyRows - begin, 1 - key_slot,
                                         tile > 1);
                    } else {
                        weight_heads(task, columns, slot, key - begin, 0, tile != 0);
                    }
                    ++tile;
                }
                if constexpr (kPipelinedRightOperands) {
                    const uint32_t last = (tile - 1) * kKeyRows;
                    weight_heads(task, end - begin - last, slot, last, (tile - 1) % kReluSlots,
                                 tile > 1);
                    // First use owns virgin storage; every subsequent use
                    // consumes its predecessor's token. Drain the final ones.
                    wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kQkEvent));
                    wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kHeadEvent));
                    wait_flag(PIPE_FIX, PIPE_M, ::event_t(kQkEvent));
                    wait_flag(PIPE_FIX, PIPE_M, ::event_t(kHeadEvent));
                    for (uint32_t relu_slot = 0; relu_slot < min(tile, uint32_t(kReluSlots));
                         ++relu_slot)
                        wait_flag(PIPE_MTE1, PIPE_FIX, ::event_t(kReluEvent + relu_slot));
                }
                // Consume each final key credit. Queue dependencies order the
                // following chunk's DMA after all prior L1 readers retire.
                for (uint32_t key_slot = 0; key_slot < min(tile, uint32_t(kKeySlots)); ++key_slot)
                    wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(kKeyEvent + key_slot));
                CrossCoreSetFlag<2, PIPE_FIX>(slot ? ScoreReady1 : ScoreReady0);
            }
        }
        // The score ring spans task boundaries so the next task's Cube work
        // can overlap the previous task's ranking. empty tasks own no token.
        for (uint32_t slot = 0; slot < min(published, uint64_t(kScoreSlots)); ++slot)
            CrossCoreWaitFlag<2, PIPE_MTE2>(slot ? ScoreCredit1 : ScoreCredit0);
        pipe_barrier(PIPE_ALL);
    }
};
} // namespace qli
