#pragma once
#include <common/cube.h>
#include <common/device.h>
#include <mhc/vector_pipeline.h>
namespace mhc {
// Two-slot GM ring between an AIC and its two AIVs. Mode2 combines the two
// AIV arrivals before waking the paired AIC; every slot has one owner.
enum RingFlag : uint16_t { InputReady = 0, InputFree = 2, ProductReady = 4, ProductFree = 6 };

class CubeProject {
    static constexpr TEventID kStatePublished = 0, kGmLoadsDrained = 0, kL1ReadsDrained = 0,
                              kMmadDrained = 0, kFixWritesDrained = 0;
    static constexpr ProjectParams p_ = kProjectPlan;
    // Slot credits use 0/1 on each independent pipe pair; FN uses 2 on
    // MTE2<->MTE1, disjoint from both activation slots. All exit waits consume
    // the outstanding credits before the named pipe drains below.
    static constexpr TEventID kL1CreditBase = 0, kL1ReadyBase = 0;
    static constexpr TEventID kOperandCreditBase = 0, kOperandReadyBase = 0;
    static constexpr TEventID accumulator_ = 0, product_ = 0;
    static constexpr TEventID fnready_ = kSlots, fnfree_ = kSlots;
    __aicore__ __cbuf__ float *activation_l1(uint32_t slot) {
        return reinterpret_cast<__cbuf__ float *>(
            uint64_t(slot * p_.tile_rows * p_.l1_tile_k * kFp32Bytes));
    }
    __aicore__ __cbuf__ float *weight_l1() {
        return reinterpret_cast<__cbuf__ float *>(
            uint64_t(kSlots * p_.tile_rows * p_.l1_tile_k * kFp32Bytes));
    }
    __aicore__ __ca__ float *activation_l0a(uint32_t slot) {
        return reinterpret_cast<__ca__ float *>(
            uint64_t(slot * p_.tile_rows * kMmadK * kFp32Bytes));
    }
    __aicore__ __cb__ float *weight_l0b(uint32_t slot) {
        return reinterpret_cast<__cb__ float *>(
            uint64_t(slot * p_.projection_stride * kMmadK * kFp32Bytes));
    }
    __aicore__ __cc__ float *accumulator_l0c() {
        return reinterpret_cast<__cc__ float *>(uint64_t(0));
    }

  public:
    __aicore__ explicit CubeProject() {
        InitSocState();
        // Each actor owns its IEEE state.
        SetSaturationFlag<SaturationMode::FLOAT>(false);
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(kL1CreditBase + slot));
            set_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandCreditBase + slot));
        }
        set_flag(PIPE_FIX, PIPE_M, ::event_t(accumulator_));
        // Entry initialization already zeros load boundary and padding.
        // Configure only the projection-specific matrix state here.
        SetHF32Mode(HF32Mode::DISABLE);
        set_l3d_rpt(uint64_t(1) << 16);
        set_fmatrix(uint64_t(p_.tile_rows) | (uint64_t(1) << 16));
        // SDK HardEvent has no S_MTE1; this exact matrix-state publication is
        // the only scalar-pipe intrinsic used by this component.
        native_device::pipe_fence<PIPE_S, PIPE_MTE1, kStatePublished>();
    }
    __aicore__ void run(GM_ADDR fn, GM_ADDR inputs, GM_ADDR products, uint32_t group, uint64_t rows) {
        const uint32_t row_tiles = (rows + p_.tile_rows - 1) / p_.tile_rows;
        auto f = gm<float>(fn), in = gm<float>(inputs), out = gm<float>(products);
        const uint32_t split = group / p_.row_groups, rg = group % p_.row_groups;
        const uint32_t tasks =
            row_tiles > rg ? (row_tiles - rg + p_.row_groups - 1) / p_.row_groups : 0;
        const uint32_t stages = (p_.panel_k + p_.l1_tile_k - 1) / p_.l1_tile_k;
        for (uint32_t panel = split; panel < p_.panel_count; panel += p_.splits) {
            const uint32_t valid = p_.input_width - panel * p_.panel_k < p_.panel_k
                                       ? p_.input_width - panel * p_.panel_k
                                       : p_.panel_k;
            {
                asc_fill_value_config fill{};
                fill.repeat = 1;
                fill.blk_num =
                    uint16_t(p_.projection_stride * p_.panel_k * kFp32Bytes / kDataBlockBytes);
                fill.dst_gap = 0;
                asc_fill_l1(weight_l1(), uint32_t(0), fill);
            };
            pipe_barrier(PIPE_MTE2);
            asc_copy_gm2l1_nd2nz(weight_l1(), f + panel * p_.panel_k, 1,
                                 uint16_t(p_.projection_columns), uint16_t(valid), 0,
                                 uint16_t(p_.input_width), uint16_t(p_.projection_stride), 1, 0);
            native_device::pipe_fence<PIPE_MTE2, PIPE_MTE1, fnready_>();
            for (uint32_t task = 0; task < tasks; ++task) {
                const uint32_t ring_slot = task % kSlots;
                CrossCoreWaitFlag<2, PIPE_MTE2>(InputReady + ring_slot);
                wait_flag(PIPE_FIX, PIPE_M, ::event_t(accumulator_));
                for (uint32_t step = 0; step <= stages; ++step) {
                    if (step < stages) {
                        const uint32_t slot = step % kSlots, start = step * p_.l1_tile_k;
                        const uint32_t depth =
                            p_.panel_k - start < p_.l1_tile_k ? p_.panel_k - start : p_.l1_tile_k;
                        wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(kL1CreditBase + slot));
                        asc_copy_gm2l1_nd2nz(activation_l1(slot),
                                             in +
                                                 (uint64_t(group) * kSlots + ring_slot) *
                                                     p_.tile_rows * p_.panel_k +
                                                 start,
                                             1, uint16_t(p_.tile_rows), uint16_t(depth), 0,
                                             uint16_t(p_.panel_k), uint16_t(p_.tile_rows), 1, 0);
                        set_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(kL1ReadyBase + slot));
                        if (step + 1 == stages && task + kSlots < tasks)
                            CrossCoreSetFlag<2, PIPE_MTE2>(InputFree + ring_slot);
                    }
                    if (step) {
                        const uint32_t part = step - 1, slot = part % kSlots,
                                       start = part * p_.l1_tile_k;
                        const uint32_t depth =
                            p_.panel_k - start < p_.l1_tile_k ? p_.panel_k - start : p_.l1_tile_k;
                        wait_flag(PIPE_MTE2, PIPE_MTE1, ::event_t(kL1ReadyBase + slot));
                        for (uint32_t chunk = 0; chunk < depth / kMmadK; ++chunk) {
                            const uint32_t os = (start / kMmadK + chunk) % kSlots;
                            wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandCreditBase + os));
                            asc_copy_l12l0a(activation_l0a(os),
                                            activation_l1(slot) + chunk * kMmadK * p_.tile_rows,
                                            kMmadK, p_.tile_rows, 0, 0, 1, 1, 1, 1, 1, 1, false,
                                            false, false, false, kMmadK);
                            asc_copy_l12l0b(
                                weight_l0b(os),
                                weight_l1() + (start + chunk * kMmadK) * p_.projection_stride, 0,
                                uint8_t(p_.projection_stride * kMmadK /
                                        (kMatrixRows * kBlockFloats)),
                                1, 0);
                            native_device::pipe_fence<PIPE_MTE1, PIPE_M>(kOperandReadyBase + os);
                            native_device::mmad(accumulator_l0c(), activation_l0a(os),
                                                weight_l0b(os), p_.tile_rows, kMmadK,
                                                p_.projection_stride, 0, part == 0 && chunk == 0);
                            set_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandCreditBase + os));
                        }
                        set_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(kL1CreditBase + slot));
                    }
                }
                native_device::pipe_fence<PIPE_M, PIPE_FIX, product_>();
                if (task >= kSlots)
                    CrossCoreWaitFlag<2, PIPE_FIX>(ProductFree + ring_slot);
                // SDK v220 row-major export owns this ND-state write and FIX barrier.
                asc_set_l0c_copy_params(1, 0, 0);
                pipe_barrier(PIPE_FIX);
                asc_copy_l0c2gm(out + (uint64_t(group) * kSlots + ring_slot) * p_.tile_rows *
                                          p_.projection_stride,
                                accumulator_l0c(), p_.projection_stride, p_.tile_rows,
                                p_.projection_stride, p_.tile_rows, 0,
                                uint64_t(QuantMode_t::NoQuant), 0, false, true);
                set_flag(PIPE_FIX, PIPE_M, ::event_t(accumulator_));
                CrossCoreSetFlag<2, PIPE_FIX>(ProductReady + ring_slot);
            }
            native_device::pipe_fence<PIPE_MTE1, PIPE_MTE2, fnfree_>();
        }
        wait_flag(PIPE_FIX, PIPE_M, ::event_t(accumulator_));
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, ::event_t(kL1CreditBase + slot));
            wait_flag(PIPE_M, PIPE_MTE1, ::event_t(kOperandCreditBase + slot));
        }
        native_device::pipe_fence<PIPE_MTE2, PIPE_S, kGmLoadsDrained>();
        native_device::pipe_fence<PIPE_MTE1, PIPE_S, kL1ReadsDrained>();
        native_device::pipe_fence<PIPE_M, PIPE_S, kMmadDrained>();
        native_device::pipe_fence<PIPE_FIX, PIPE_S, kFixWritesDrained>();
    }
};
__aicore__ inline void project_vector(GM_ADDR residual, GM_ADDR projection, GM_ADDR squares,
                                      GM_ADDR inputs, GM_ADDR products, uint32_t group,
                                      uint32_t lane, uint64_t rows, uint64_t padded_rows) {
    constexpr const ProjectParams &p = kProjectPlan;
    const uint32_t row_tiles = (rows + p.tile_rows - 1) / p.tile_rows;
    InitSocState();
    // arch2201 InitSocState leaves FLOAT saturation unchanged.
    SetSaturationFlag<SaturationMode::FLOAT>(false);
    VectorEvents ev;
    const uint32_t half = p.tile_rows / 2, k = p.panel_k, n = p.projection_stride;
    const ProjectionVectorLayout layout(p.tile_rows, k, n);
    const uint32_t vr = layout.rows;
    auto raw = ub_ptr<bfloat16_t>(layout.raw);
    auto value = ub_ptr<float>(layout.value), squared = ub_ptr<float>(layout.squared);
    auto scratch = ub_ptr<float>(layout.scratch);
    auto sums = ub_ptr<float>(layout.sums);
    auto acc = ub_ptr<float>(layout.accumulator), product = ub_ptr<float>(layout.product);
    auto stat = ub_ptr<float>(layout.stat);
    auto r = gm<bfloat16_t>(residual);
    auto proj = gm<float>(projection), sq = gm<float>(squares), in = gm<float>(inputs),
         out = gm<float>(products);
    const uint32_t split = group / p.row_groups, rg = group % p.row_groups;
    const uint32_t tasks =
        row_tiles > rg ? (row_tiles - rg + p.row_groups - 1) / p.row_groups : 0;
    for (uint32_t panel = split; panel < p.panel_count; panel += p.splits) {
        const uint32_t valid_k = p.input_width - panel * k < k ? p.input_width - panel * k : k;
        for (uint32_t step = 0; step < tasks + kSlots; ++step) {
            if (step >= kSlots) {
                const uint32_t task = step - kSlots, slot = task % kSlots;
                const uint64_t row = uint64_t(rg + task * p.row_groups) * p.tile_rows + lane * half;
                CrossCoreWaitFlag<2, PIPE_MTE2>(ProductReady + slot);
                native_device::pipe_fence<PIPE_V, PIPE_MTE2, VectorEvents::raw_free>();
                asc_copy_gm2ub_align(
                    product,
                    (out + ((uint64_t(group) * kSlots + slot) * p.tile_rows * n + lane * half * n)),
                    1, half * n * sizeof(*(product)), 0, 0, 0, 0);
                if (panel == split) {
                    {
                        asc_duplicate(acc, 0.0f, half * n);
                        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
                    };
                    pipe_barrier(PIPE_V);
                } else
                    asc_copy_gm2ub_align(acc,
                                         (proj + ((uint64_t(split) * padded_rows + row) * n)), 1,
                                         half * n * sizeof(*(acc)), 0, 0, 0, 0);
                native_device::pipe_fence<PIPE_MTE2, PIPE_V, VectorEvents::loaded>();
                if (task + kSlots < tasks)
                    CrossCoreSetFlag<2, PIPE_MTE2>(ProductFree + slot);
                {
                    asc_add(acc, acc, product, half * n);
                    asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
                };
                pipe_barrier(PIPE_V);
                native_device::pipe_fence<PIPE_V, PIPE_MTE3, VectorEvents::store_ready>();
                asc_copy_ub2gm_align((proj + ((uint64_t(split) * padded_rows + row) * n)), acc, 1,
                                     half * n * sizeof(*(acc)), 0, 0, 0, 0);
                native_device::pipe_fence<PIPE_MTE3, PIPE_V, VectorEvents::stored>();
            }
            if (step < tasks) {
                const uint32_t slot = step % kSlots;
                const uint64_t row = uint64_t(rg + step * p.row_groups) * p.tile_rows + lane * half;
                if (step >= kSlots)
                    CrossCoreWaitFlag<2, PIPE_MTE3>(InputFree + slot);
                if (panel == split) {
                    {
                        asc_duplicate(sums, 0.0f, round_up(half, kBlockFloats));
                        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
                    };
                    pipe_barrier(PIPE_V);
                } else {
                    native_device::pipe_fence<PIPE_V, PIPE_MTE2, VectorEvents::raw_free>();
                    asc_copy_gm2ub_align(sums, (sq + (uint64_t(split) * padded_rows + row)), 1,
                                         half * sizeof(*(sums)), 0, 0, 0, 0);
                    native_device::pipe_fence<PIPE_MTE2, PIPE_V, VectorEvents::loaded>();
                }
                for (uint32_t first = 0; first < half; first += vr) {
                    const uint64_t token = row + first;
                    const uint32_t valid_rows =
                        token >= rows ? 0 : (rows - token < vr ? rows - token : vr);
                    if (first)
                        wait_flag(PIPE_V, PIPE_MTE2, ::event_t(ev.raw_reuse[0]));
                    else {
                        native_device::pipe_fence<PIPE_V, PIPE_MTE2, VectorEvents::raw_free>();
                    }
                    if (valid_rows < vr || valid_k < k) {
                        {
                            asc_duplicate(reinterpret_cast<__ubuf__ uint16_t *>(raw), uint16_t(0),
                                          vr * k);
                            asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
                        };
                        pipe_barrier(PIPE_V);
                        native_device::pipe_fence<PIPE_V, PIPE_MTE2, VectorEvents::raw_free>();
                    }
                    if (valid_rows)
                        asc_copy_gm2ub(
                            raw, (r + (token * p.input_width + panel * k)), uint16_t(valid_rows),
                            uint16_t(valid_k * kBf16Bytes / kDataBlockBytes),
                            uint16_t((p.input_width - valid_k) * kBf16Bytes / kDataBlockBytes),
                            uint16_t((k - valid_k) * kBf16Bytes / kDataBlockBytes));
                    native_device::pipe_fence<PIPE_MTE2, PIPE_V, VectorEvents::loaded>();
                    if (first)
                        wait_flag(PIPE_MTE3, PIPE_V, ::event_t(ev.packed_reuse[0]));
                    {
                        asc_bfloat162float(value, raw, vr * k);
                        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
                    };
                    if (first + vr < half)
                        set_flag(PIPE_V, PIPE_MTE2, ::event_t(ev.raw_reuse[0]));
                    pipe_barrier(PIPE_V);
                    native_device::pipe_fence<PIPE_V, PIPE_MTE3, VectorEvents::store_ready>();
                    asc_copy_ub2gm_align(
                        (in + ((uint64_t(group) * kSlots + slot) * p.tile_rows * k +
                               (lane * half + first) * k)),
                        value, 1, vr * k * sizeof(*(value)), 0, 0, 0, 0);
                    if (first + vr == half)
                        CrossCoreSetFlag<2, PIPE_MTE3>(InputReady + slot);
                    else
                        set_flag(PIPE_MTE3, PIPE_V, ::event_t(ev.packed_reuse[0]));
                    {
                        asc_mul(squared, value, value, vr * k);
                        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
                    };
                    pipe_barrier(PIPE_V);
                    panel_sum(stat, squared, scratch, vr, k);
                    {
                        asc_add((sums + (first)), (sums + (first)), stat, vr);
                        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
                    };
                    pipe_barrier(PIPE_V);
                }
                native_device::pipe_fence<PIPE_V, PIPE_MTE3, VectorEvents::store_ready>();
                asc_copy_ub2gm_align((sq + (uint64_t(split) * padded_rows + row)), sums, 1,
                                     half * sizeof(*(sums)), 0, 0, 0, 0);
                native_device::pipe_fence<PIPE_MTE3, PIPE_V, VectorEvents::stored>();
            }
        }
    }
    {
        native_device::pipe_fence<PIPE_MTE2, PIPE_S, VectorEvents::load_drained>();
        native_device::pipe_fence<PIPE_MTE3, PIPE_S, VectorEvents::store_drained>();
        native_device::pipe_fence<PIPE_V, PIPE_S, VectorEvents::scalar_read>();
    }
}
} // namespace mhc
