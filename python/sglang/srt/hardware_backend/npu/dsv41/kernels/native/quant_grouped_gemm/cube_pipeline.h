#pragma once
#include <common/cube.h>
#include <common/device.h>
#include <cstdint>
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <quant_grouped_gemm/layout.h>
#include <common/grouped_schedule.h>

namespace quant_gmm {
using grouped_gemm::Tile;
using namespace AscendC;

// Two L1 panels and two L0 operands overlap MTE2 -> MTE1 -> M. AIC reads the
// AIV-produced GM panel only after both AIVs publish it. L1 retirement returns
// the GM slot. MMAD/Fixpipe unit flags 2/3 protect the single L0C accumulator;
// changing its M footprint additionally retires FIX before the next import.
template <class P> class CubePipeline {
    using Shape = PanelShape<P>;
    __gm__ half *activation_, *output_, *workspace_;
    int initialized_rows0_ = 0, initialized_rows1_ = 0;

    __aicore__ inline __cbuf__ half *activation_l1(int slot) {
        return reinterpret_cast<__cbuf__ half *>(uint64_t(slot * P::A1Bytes));
    }
    __aicore__ inline __cbuf__ half *weight_l1(int slot) {
        return reinterpret_cast<__cbuf__ half *>(uint64_t(kSlots * P::A1Bytes + slot * P::B1Bytes));
    }
    __aicore__ inline static void set_left_matrix_rows(int rows) {
        // The legacy register packet is W | H<<16, namely W=rows, H=1.
        // Preserve that flat NZ import; swapping H/W can pass numerics while
        // greatly increasing MTE1 service time.
        set_fmatrix(uint64_t(rows) | (uint64_t(1) << 16));
    }
    __aicore__ inline void initialize() {
        asc_set_l0c_copy_params(1, 0, 0);
        native_device::pipe_fence<PIPE_S, PIPE_FIX, kFixStateEvent>();
        set_l3d_rpt(uint64_t(1) << 16);
        // Entry InitSocState() already establishes zero padding and boundary.
        set_left_matrix_rows(P::Rows);
        set_fmatrix_b(uint64_t(Shape::step_k) | (uint64_t(1) << 16));
        // Register writes need Scalar->MTE1 ordering; these pipe pairs have
        // no HardEvent enumerator, so issue their explicit pipe/event instructions.
        native_device::pipe_fence<PIPE_S, PIPE_MTE1, kMatrixStateEvent>();
    }
    __aicore__ inline void compute_chunk(int rows, int generation, int known, int panel, int chunk,
                                         int depth) {
        const int slot = (generation & (kSlots - 1));
        const int dot = generation * Shape::chunks + chunk;
        const int operand_slot = (dot & (kSlots - 1));
        auto activation_l0a = reinterpret_cast<__ca__ half *>(uint64_t(operand_slot * P::A0Bytes));
        auto weight_l0b = reinterpret_cast<__cb__ half *>(uint64_t(operand_slot * P::B0Bytes));
        auto accumulator_l0c = reinterpret_cast<__cc__ float *>(uint64_t(0));
        if (dot >= kSlots)
            wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandReuseEvent + operand_slot));
        if (rows == kFractalRows)
            asc_copy_l12l0a(activation_l0a, activation_l1(slot) + chunk * P::DotK * rows,
                            0, uint8_t(depth / kFractalRows), 1, 0);
        else
            native_device::load_a16_3d(activation_l0a, activation_l1(slot) + chunk * P::DotK * rows,
                                       rows, depth);
        if (chunk + 1 == Shape::chunks && (known < 0 || generation + kSlots < known))
            set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot));
        if (!chunk)
            wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot + kRightOperandEventBase));
        native_device::load_b16_transposed(
            weight_l0b, weight_l1(slot) + chunk * P::DotK * kFractalRows, P::Columns, depth);
        if (chunk + 1 == Shape::chunks && (known < 0 || generation + kSlots < known))
            set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot + kRightOperandEventBase));
        native_device::pipe_fence<PIPE_MTE1, PIPE_M>(operand_slot);
        const uint8_t unit_flag = panel + 1 == Shape::panels && chunk + 1 == Shape::chunks
                                      ? kAccumulatorPublish
                                      : kAccumulatorContinue;
        native_device::mmad(accumulator_l0c, activation_l0a, weight_l0b, rows, depth, P::Columns,
                            unit_flag, panel == 0 && chunk == 0);
        set_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandReuseEvent + operand_slot));
    }
    __aicore__ inline void compute_tile(const Tile &tile, int generation, int known,
                                        int &previous_rows, int panel) {
        const int slot = (generation & (kSlots - 1));
        const int rows = PanelShape<P>::matrix_rows(tile.valid);
        if (!panel && generation > 0 && rows != previous_rows) {
            native_device::pipe_fence<PIPE_FIX, PIPE_MTE1, kFootprintRetiredEvent>();
        }
        if (!panel && rows > kFractalRows && rows != previous_rows) {
            native_device::pipe_fence<PIPE_MTE1, PIPE_S, kMatrixStateEvent>();
            set_left_matrix_rows(rows);
            native_device::pipe_fence<PIPE_S, PIPE_MTE1, kMatrixStateEvent>();
        }
        if (!panel)
            previous_rows = rows;
        wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot));
        const int full_chunks = Shape::step_k / P::DotK;
        for (int chunk = 0; chunk < full_chunks; ++chunk)
            compute_chunk(rows, generation, known, panel, chunk, P::DotK);
        if (Shape::step_k % P::DotK)
            compute_chunk(rows, generation, known, panel, full_chunks, Shape::step_k % P::DotK);
        if (panel + 1 == Shape::panels) {
            // UnitFlag publishes this accumulator; FIX retirement remains explicit.
            auto accumulator_l0c = reinterpret_cast<__cc__ float *>(uint64_t(0));
            asc_copy_l0c2gm(output_ + int64_t(tile.row) * Shape::n + tile.col, accumulator_l0c,
                            P::Columns, tile.valid, Shape::n, rows, kAccumulatorPublish,
                            uint64_t(QuantMode_t::F322F16), 0, false, true);
        }
    }
    __aicore__ inline void load_panel(const Tile &tile, int generation, int known, int core,
                                      int panel) {
        const int slot = (generation & (kSlots - 1));
        if (generation >= kSlots)
            wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot));
        const int rows = PanelShape<P>::matrix_rows(tile.valid);
        // Preserve the original first-use initialization of each L1 slot.
        // A larger footprint needs its inactive rows initialized once.
        const int initialized = slot == 0 ? initialized_rows0_ : initialized_rows1_;
        if (initialized < rows) {
            if (tile.valid < rows) {
                asc_fill_value_config fill{};
                fill.repeat = 1;
                fill.blk_num = rows * Shape::step_k * sizeof(half) / kDmaBytes;
                asc_fill_l1(activation_l1(slot), uint32_t(0), fill);
                pipe_barrier(PIPE_MTE2);
            }
            if (slot == 0)
                initialized_rows0_ = rows;
            else
                initialized_rows1_ = rows;
        }
        asc_copy_gm2l1_nd2nz(activation_l1(slot),
                             activation_ + int64_t(tile.row) * Shape::k + panel * Shape::step_k, 1,
                             uint16_t(tile.valid), uint16_t(Shape::step_k), 0, uint16_t(Shape::k),
                             uint16_t(rows), 1, 0);
        set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot));
        if (generation >= kSlots)
            wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot + kRightOperandEventBase));
        if (slot == 0)
            CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE2>(kPanel0Ready);
        else
            CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE2>(kPanel1Ready);
        // AIV emits the complete native NZ panel; no weight ND2NZ remains.
        const int64_t source = (int64_t(core) * kSlots + slot) *
            Shape::workspace_panel_bytes / sizeof(half);
        asc_copy_gm2l1(weight_l1(slot), workspace_ + source,
                      uint32_t(Shape::workspace_panel_bytes));
        set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot + kRightOperandEventBase));
        if (known < 0 || generation + kSlots < known) {
            if (slot == 0)
                CrossCoreSetFlag<kGroupFlagMode, PIPE_MTE2>(kPanel0Free);
            else
                CrossCoreSetFlag<kGroupFlagMode, PIPE_MTE2>(kPanel1Free);
        }
    }
    __aicore__ inline void drain(int known, int count, int dots) {
        for (int slot = 0; slot < kSlots; ++slot) {
            if (known < 0 && count > slot) {
                wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot));
                wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot + kRightOperandEventBase));
            }
            if (dots > slot)
                wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandReuseEvent + slot));
        }
        // Keep the original four independent exit pairs. They include the
        // final credit consumers on MTE2/MTE1, not just the exported result.
        native_device::pipe_fence<PIPE_MTE2, PIPE_S, kRetirementEvent>();
        native_device::pipe_fence<PIPE_MTE1, PIPE_S, kRetirementEvent>();
        native_device::pipe_fence<PIPE_M, PIPE_S, kRetirementEvent>();
        native_device::pipe_fence<PIPE_FIX, PIPE_S, kRetirementEvent>();
    }

  public:
    __aicore__ inline void run(GM_ADDR activation, GM_ADDR routing, GM_ADDR output,
                               GM_ADDR workspace, int experts, int cores) {
        activation_ = reinterpret_cast<__gm__ half *>(activation);
        output_ = reinterpret_cast<__gm__ half *>(output);
        workspace_ = reinterpret_cast<__gm__ half *>(workspace);
        const int core = GetBlockIdx();
        grouped_gemm::Schedule<P::Rows, P::Columns> schedule(routing, experts, Shape::panels, Shape::n, core, cores);
        Tile next{}, current{};
        if (schedule.next(core, next)) {
            // Empty actors own no flags or local buffers to retire.
            initialize();
            const int known = schedule.known_steps(core, cores);
            int count = known, dots = 0, previous_rows = P::Rows;
            // Original one-panel lookahead: compute step-1, then import step.
            for (int step = 0; count < 0 || step <= count; ++step) {
                if (step) {
                    const int generation = step - 1;
                    compute_tile(current, generation, known, previous_rows,
                                 generation % Shape::panels);
                    dots += Shape::chunks;
                }
                if (count < 0 || step < count) {
                    const int panel = step % Shape::panels;
                    if (!panel && !schedule.next(core + step / Shape::panels * cores, next))
                        count = step;
                    else {
                        load_panel(next, step, known, core, panel);
                        current = next;
                    }
                }
            }
            drain(known, count, dots);
        }
    }
};
} // namespace quant_gmm
