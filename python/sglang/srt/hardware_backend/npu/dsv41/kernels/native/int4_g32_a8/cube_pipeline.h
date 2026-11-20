#pragma once
#include <common/cube.h>
#include <common/device.h>
#include <int4_g32_a8/layout.h>

namespace int4_g32_a8 {
__aicore__ inline void cube(const Inputs &args, const Tile &tile, int part, int generation, int job,
                            __gm__ int8_t *bridge, int live_rows, int known, bool same_activation) {
    const int slot = generation & 1;
    auto left = reinterpret_cast<__cbuf__ int8_t *>(
        uint64_t((keep_activation ? part : slot) * rows * panel));
    auto right = reinterpret_cast<__cbuf__ int8_t *>(
        uint64_t(activation_slots * rows * panel + slot * weight_bytes));
    auto acc = reinterpret_cast<__cc__ int32_t *>(uint64_t(0));
    if (generation >= slots)
        wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot));
    if (!keep_activation || !same_activation) {
        asc_copy_gm2l1_nd2nz(left,
                             reinterpret_cast<__gm__ int8_t *>(args.activation) +
                                 int64_t(tile.row) * depth + part * panel,
                             1, uint16_t(tile.valid), uint16_t(panel), 0, uint16_t(depth),
                             uint16_t(live_rows), 1, 0);
        set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot));
    }
    CrossCoreWaitFlag<2, PIPE_MTE2>(slot);
    asc_copy_gm2l1(right, bridge + slot * weight_bytes, uint32_t(weight_bytes));
    if (known < 0 || generation + slots < known)
        CrossCoreSetFlag<2, PIPE_MTE2>(2 + slot);
    set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(2 + slot));
    if (!keep_activation || !same_activation)
        wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(slot));
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int serial = generation * chunks + chunk, operand = serial & 1;
        const int live = chunk + 1 == chunks ? panel - chunk * dot_k : dot_k;
        auto left0 = reinterpret_cast<__ca__ int8_t *>(uint64_t(operand * rows * dot_k));
        auto right0 = reinterpret_cast<__cb__ int8_t *>(uint64_t(operand * columns * dot_k));
        if (serial >= slots)
            wait_flag(PIPE_M, PIPE_MTE1, ::event_t(operand));
        auto src = left + chunk * live_rows * dot_k;
        if (live_rows == fractal_rows)
            asc_copy_l12l0a(left0, src, 0, uint8_t(live / group), 1, 0);
        else
            asc_copy_l12l0a(left0, src, live, live_rows, 0, 0, 1, 1, 1, 1, 1, 1, false, false,
                            false, false, live);
        if (!chunk)
            wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(2 + slot));
        asc_copy_l12l0b_trans(right0, right + chunk * columns * dot_k, 0,
                              uint8_t(columns * live / 1024), 1, 1, false, 0);
        if (chunk + 1 == chunks && (known < 0 || generation + slots < known))
            set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(slot));
        native_device::pipe_fence<PIPE_MTE1, PIPE_M>(operand);
        const uint8_t unit = part + 1 == panels && chunk + 1 == chunks ? 3 : 2;
        native_device::mmad(acc, left0, right0, live_rows, live, columns, unit, !part && !chunk);
        if (known < 0 || serial + slots < known * chunks)
            set_flag(PIPE_M, PIPE_MTE1, ::event_t(operand));
    }
    if (part + 1 == panels) {
        auto result =
            reinterpret_cast<__gm__ int32_t *>(bridge + result_offset + (job & 1) * result_bytes);
        const int fix_rows = keep_activation && tile.valid > rows / 2 ? live_rows : tile.valid;
        asc_copy_l0c2gm(result, acc, uint16_t(columns), uint16_t(fix_rows), columns, live_rows, 3,
                        uint64_t(QuantMode_t::NoQuant), 0, false, true);
        CrossCoreSetFlag<2, PIPE_FIX>(4 + (job & 1));
    }
}
} // namespace int4_g32_a8
