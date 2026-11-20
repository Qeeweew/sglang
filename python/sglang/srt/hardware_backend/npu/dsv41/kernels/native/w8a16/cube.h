#pragma once
#include <common/cube.h>
#include <common/device.h>
#include <w8a16/layout.h>

namespace w8a16 {
class CubePipeline {
    __gm__ Element *input_, *workspace_;
#if W8A16_OUTPUT_BF16
    using Output = bfloat16_t;
#else
    using Output = half;
#endif
    __gm__ Output *output_;
    int previous_rows_ = kRows;
    int initialized_valid_[kSlots] = {-1, -1};
    __aicore__ inline static __cbuf__ Element *activation_l1(int slot) {
        return reinterpret_cast<__cbuf__ Element *>(uint64_t(slot * kMemory.a1_bytes));
    }
    __aicore__ inline static __cbuf__ Element *weight_l1(int slot) {
        return reinterpret_cast<__cbuf__ Element *>(uint64_t(kSlots * kMemory.a1_bytes + slot * kMemory.b1_bytes));
    }
    __aicore__ inline void load_panel(const Tile &tile, int generation, int count, int core) {
        const int slot = generation % kSlots, panel = generation % kPanels;
        if (generation >= kSlots) wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot));
        if (tile.valid != tile.matrix_rows && initialized_valid_[slot] != tile.valid) {
            // Only tail rows read by MMAD need a defined zero value. ND2NZ's
            // valid row copy leaves them unchanged; initialize the compact span.
            asc_fill_value_config fill{};
            fill.repeat = 1;
            fill.blk_num = tile.matrix_rows * kPanelK * 2 / 32;
            asc_fill_l1(reinterpret_cast<__cbuf__ uint32_t *>(activation_l1(slot)), uint32_t(0), fill);
            pipe_barrier(PIPE_MTE2);
        }
        initialized_valid_[slot] = tile.valid;
        asc_copy_gm2l1_nd2nz(activation_l1(slot),
            input_ + (tile.row * kBatches + tile.batch) * kLogicalK + panel * kPanelK,
            1, uint16_t(tile.valid), uint16_t(kPanelK), 0, uint16_t(kBatches * kLogicalK),
            uint16_t(tile.matrix_rows), 1, 0);
        set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot));
        if (generation >= kSlots)
            wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot + kWeightPanelEventBase));
        if (slot == 0) CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE2>(kPanel0Ready);
        else CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE2>(kPanel1Ready);
        asc_copy_gm2l1(weight_l1(slot), workspace_ + (int64_t(core) * kSlots + slot) * kPanelK * kColumns,
                       uint32_t(kMemory.b1_bytes));
        set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot + kWeightPanelEventBase));
        if (generation + kSlots < count) {
            if (slot == 0) CrossCoreSetFlag<kGroupFlagMode, PIPE_MTE2>(kPanel0Free);
            else CrossCoreSetFlag<kGroupFlagMode, PIPE_MTE2>(kPanel1Free);
        }
    }
    __aicore__ inline void compute_panel(const Tile &tile, int generation, int count) {
        const int slot = generation % kSlots, panel = generation % kPanels;
        const int rows = tile.matrix_rows;
        if (!panel && generation > 0 && rows != previous_rows_) {
            native_device::pipe_fence<PIPE_FIX, PIPE_MTE1, kFootprintRetiredEvent>();
        }
        if (!panel && rows != previous_rows_) {
            native_device::pipe_fence<PIPE_MTE1, PIPE_S, kStateEvent>();
            set_fmatrix(uint64_t(rows) | (uint64_t(1) << 16));
            native_device::pipe_fence<PIPE_S, PIPE_MTE1, kStateEvent>();
            previous_rows_ = rows;
        }
        wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot));
        constexpr int chunks = kPanelK / kMemory.dot_k;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int dot = generation * chunks + chunk, operand_slot = dot % kSlots;
            auto left = reinterpret_cast<__ca__ Element *>(uint64_t(operand_slot * kMemory.a0_bytes));
            auto right = reinterpret_cast<__cb__ Element *>(uint64_t(operand_slot * kMemory.b0_bytes));
            auto accumulator = reinterpret_cast<__cc__ float *>(uint64_t(0));
            if (dot >= kSlots) wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandReuseEventBase + operand_slot));
            if (rows == kC0)
                asc_copy_l12l0a(left, activation_l1(slot) + chunk * kMemory.dot_k * rows,
                                0, uint8_t(kMemory.dot_k / kC0), 1, 0);
            else
                native_device::load_a16_3d(left, activation_l1(slot) + chunk * kMemory.dot_k * rows,
                                           rows, kMemory.dot_k);
            if (chunk + 1 == chunks && generation + kSlots < count)
                set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot));
            if (!chunk)
                wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot + kWeightPanelEventBase));
            native_device::load_b16_transposed(right, weight_l1(slot) + chunk * kMemory.dot_k * kC0,
                                               kColumns, kMemory.dot_k);
            if (chunk + 1 == chunks && generation + kSlots < count)
                set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot + kWeightPanelEventBase));
            native_device::pipe_fence<PIPE_MTE1, PIPE_M>(operand_slot);
            const uint8_t unit_flag = panel + 1 == kPanels && chunk + 1 == chunks
                                          ? kAccumulatorPublish : kAccumulatorContinue;
            native_device::mmad(accumulator, left, right, rows, kMemory.dot_k, kColumns, unit_flag,
                                panel == 0 && chunk == 0);
            set_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandReuseEventBase + operand_slot));
        }
        if (panel + 1 == kPanels) {
            constexpr auto quant = W8A16_OUTPUT_BF16 ? QuantMode_t::F322BF16 : QuantMode_t::F322F16;
            copy_matrix_cc_to_gm(output_ + (tile.row * kBatches + tile.batch) * kLogicalN + tile.column,
                reinterpret_cast<__cc__ float *>(uint64_t(0)), 0, uint16_t(kColumns), uint16_t(tile.valid),
                kBatches * kLogicalN, rows, kAccumulatorPublish, quant, 0, false, true);
        }
    }
  public:
    __aicore__ inline void run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, int64_t rows) {
        input_ = reinterpret_cast<__gm__ Element *>(input);
        output_ = reinterpret_cast<__gm__ Output *>(output);
        workspace_ = reinterpret_cast<__gm__ Element *>(workspace);
        asc_set_l0c_copy_params(1, 0, 0);
        native_device::pipe_fence<PIPE_S, PIPE_FIX, kAccumulatorEvent>();
        set_l3d_rpt(uint64_t(1) << 16);
        set_fmatrix(uint64_t(kRows) | (uint64_t(1) << 16));
        set_fmatrix_b(uint64_t(kPanelK) | (uint64_t(1) << 16));
        native_device::pipe_fence<PIPE_S, PIPE_MTE1, kStateEvent>();
        const int core = GetBlockIdx();
        const int64_t jobs = (rows + kRows - 1) / kRows * (kLogicalN / kColumns) * kBatches;
        const int count = int((jobs - core + kWorkers - 1) / kWorkers) * kPanels;
        Tile current{};
        for (int step = 0; step <= count; ++step) {
            if (step) compute_panel(current, step - 1, count);
            if (step < count) {
                current = coordinates(core + int64_t(step / kPanels) * kWorkers, rows);
                load_panel(current, step, count, core);
            }
        }
        constexpr int chunks = kPanelK / kMemory.dot_k;
        for (int slot = 0; slot < kSlots; ++slot)
            if (count * chunks > slot)
                wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandReuseEventBase + slot));
        native_device::pipe_fence<PIPE_MTE2, PIPE_S, kDrainEvent>();
        native_device::pipe_fence<PIPE_MTE1, PIPE_S, kDrainEvent>();
        native_device::pipe_fence<PIPE_M, PIPE_S, kDrainEvent>();
        native_device::pipe_fence<PIPE_FIX, PIPE_S, kDrainEvent>();
    }
};
} // namespace w8a16
