#pragma once
#include <common/device.h>
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <quant_grouped_gemm/layout.h>
#include <common/grouped_schedule.h>

namespace quant_gmm {
using native_device::ub_ptr;
using grouped_gemm::Tile;
using namespace AscendC;
// Native FP16 NZ stream: [owned N16 blocks, StepK, 16]. Input panel credits
// and output byte-chunk credits are independent. Both AIVs publish a panel;
// AIC returns its GM slot only after the ordinary GM->L1 copy completes.
template <class P> class DecodePipeline {
    using Shape = PanelShape<P>;
    __gm__ uint8_t *packed_, *scales_;
    __gm__ half *workspace_;
    __aicore__ inline void initialize() {
        asc_duplicate(ub_ptr<int16_t>(P::ThresholdOffset), kRankCorrectionBits,
                      uint32_t(kHalfVectorElements));
        asc_sync_pipe(PIPE_V);
    }
    __aicore__ inline void unpack_scales(int slot) {
        asc_sync_pipe(PIPE_V);
        auto factors = ub_ptr<half>(P::ScaleOutputOffset);
        auto raw = ub_ptr<uint16_t>(P::ScaleOffset + slot * P::ScaleSlotBytes);
        constexpr int scale_elements = P::LocalBlocks * P::ScaleGroups * P::C0Elements;
        // A compact N16 scale row is16 bytes, while vector operands need
        // 32-byte addresses. Convert byte values in a temporary prefix,
        // then reuse the retired raw slot for aligned FP16 scale rows.
        if constexpr (Shape::step_k == Shape::k || P::ScaleGroups % 2 == 0)
            asc_uint82half(factors, reinterpret_cast<__ubuf__ uint8_t *>(raw), uint32_t(scale_elements));
        else
            for (int block = 0; block < P::LocalBlocks; ++block)
                asc_uint82half(factors + block * P::ScaleGroups * P::C0Elements,
                    reinterpret_cast<__ubuf__ uint8_t *>(raw) + block * P::ScaleBlockBytes,
                    uint32_t(P::ScaleGroups * P::C0Elements));
        asc_sync_pipe(PIPE_V);
        asc_half2int16_rn(reinterpret_cast<__ubuf__ int16_t *>(raw), factors, uint32_t(scale_elements));
        asc_sync_pipe(PIPE_V);
        asc_shiftleft(raw, raw, uint16_t(8), uint32_t(scale_elements));
        asc_sync_pipe(PIPE_V);
        asc_leakyrelu(reinterpret_cast<__ubuf__ half *>(raw), reinterpret_cast<__ubuf__ half *>(raw),
                      half(kTaggedScaleCorrection), uint32_t(scale_elements));
        asc_sync_pipe(PIPE_V);
        // Keep one aligned16-half scale row per quantization group. Mul's
        // zero source-block stride broadcasts it without an expanded tensor.
        asc_add_scalar(reinterpret_cast<__ubuf__ int16_t *>(factors),
                       reinterpret_cast<__ubuf__ int16_t *>(raw), int16_t(0), uint32_t(scale_elements));
    }
    __aicore__ inline void decode_micro(__ubuf__ half *decoded, __ubuf__ uint8_t *raw,
                                        int first, int elements, bool release, int slot) {
        asc_sync_pipe(PIPE_V);
        asc_set_mask_norm();
        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
        const uint8_t repeats = elements / kHalfVectorElements;
        asc_int42half(decoded, reinterpret_cast<__ubuf__ int4b_t *>(raw), repeats, 1, 1, 8, 2);
        if (release) set_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
        asc_sync_pipe(PIPE_V);
        auto absolute = ub_ptr<int16_t>(P::AbsoluteOffset);
        asc_abs(reinterpret_cast<__ubuf__ half *>(absolute), decoded, repeats, 1, 1, 8, 8);
        asc_sync_pipe(PIPE_V);
        asc_sub_relu(absolute, absolute, ub_ptr<int16_t>(P::ThresholdOffset), repeats,
                     1, 1, 1, 8, 8, 0);
        asc_sync_pipe(PIPE_V);
        auto bits = reinterpret_cast<__ubuf__ int16_t *>(decoded);
        asc_add(bits, bits, absolute, repeats, 1, 1, 1, 8, 8, 8);
        asc_sync_pipe(PIPE_V);
#pragma unroll
        for (int group = 0; group < P::MicroElements / P::AtomElements; ++group)
            if (group < elements / P::AtomElements)
                asc_mul(decoded + group * P::AtomElements, decoded + group * P::AtomElements,
                    ub_ptr<half>(P::ScaleOutputOffset) +
                        (first / P::AtomElements + group) * P::C0Elements,
                    uint8_t(P::AtomElements / kHalfVectorElements), 1, 1, 0, 8, 8, 0);
    }
    __aicore__ inline void load_weights(const Tile &tile, int generation, int lane) {
        const int slot = generation & 1;
        if (generation >= kSlots) {
            wait_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
            wait_flag(PIPE_V, PIPE_MTE2, ::event_t(slot + kScaleEventBase));
        }
        const int column = tile.col + lane * (P::Columns / kVectorLanesPerCube);
        const int start = generation % Shape::panels * Shape::step_k;
        const int64_t code_offset = int64_t(tile.expert) * Shape::k * Shape::n / kPackedPerByte +
            int64_t(column) * Shape::k / kPackedPerByte + start * P::C0Elements / kPackedPerByte;
        const int64_t scale_offset = (int64_t(tile.expert) * Shape::k * Shape::n / kScaleGroup +
            int64_t(column) * (Shape::k / kScaleGroup) + start / kScaleGroup * P::C0Elements) * P::ScaleElementBytes;
        if constexpr (Shape::step_k == Shape::k)
            asc_copy_gm2ub_align(ub_ptr<uint8_t>(P::ScaleOffset + slot * P::ScaleSlotBytes),
                scales_ + scale_offset, uint16_t(1),
                uint32_t(P::LocalBlocks * P::ScaleGroups * P::C0Elements), 0, 0, 0, 0);
        else
            asc_copy_gm2ub_align(ub_ptr<uint8_t>(P::ScaleOffset + slot * P::ScaleSlotBytes),
                scales_ + scale_offset, uint16_t(P::LocalBlocks),
                uint32_t(P::ScaleGroups * P::C0Elements * P::ScaleElementBytes), 0, 0,
                uint32_t((Shape::k - Shape::step_k) / kScaleGroup * P::C0Elements * P::ScaleElementBytes), 0);
        set_flag(PIPE_MTE2, PIPE_V, ::event_t(slot + kScaleEventBase));
        asc_copy_gm2ub(ub_ptr<uint8_t>(slot * P::CodeCapacity), packed_ + code_offset,
            uint16_t(P::LocalBlocks), uint16_t(Shape::step_k * P::C0Elements / kPackedPerByte / kDmaBytes),
            uint16_t((Shape::k - Shape::step_k) * P::C0Elements / kPackedPerByte / kDmaBytes), 0);
        set_flag(PIPE_MTE2, PIPE_V, ::event_t(slot));
    }
    __aicore__ inline void compute_block(int block, int known) {
        const int generation = block / P::Blocks, inner = block % P::Blocks;
        const int slot = generation & 1, output_slot = block & 1;
        const int first = inner * P::ChunkElements;
        const int elements = min(P::LocalElements - first, P::ChunkElements);
        const bool future = known < 0 || generation + kSlots < known;
        if (block >= kSlots) wait_flag(PIPE_MTE3, PIPE_V, ::event_t(output_slot));
        if (!inner) {
            wait_flag(PIPE_MTE2, PIPE_V, ::event_t(slot + kScaleEventBase));
            unpack_scales(slot);
            if (future) set_flag(PIPE_V, PIPE_MTE2, ::event_t(slot + kScaleEventBase));
            wait_flag(PIPE_MTE2, PIPE_V, ::event_t(slot));
        }
        auto decoded = ub_ptr<half>(P::DecodedOffset + output_slot * P::DecodeCapacity);
        for (int local = 0; local < elements; local += P::MicroElements) {
            const int count = min(P::MicroElements, elements - local);
            decode_micro(decoded + local, ub_ptr<uint8_t>(slot * P::CodeCapacity + (first + local) / 2),
                first + local, count, future && inner + 1 == P::Blocks && local + count == elements, slot);
        }
        set_flag(PIPE_V, PIPE_MTE3, ::event_t(output_slot));
    }
    __aicore__ inline void store_block(int block, int count, int core, int lane) {
        const int generation = block / P::Blocks, inner = block % P::Blocks;
        const int slot = generation & 1, output_slot = block & 1;
        const int first = inner * P::ChunkElements;
        const int elements = min(P::LocalElements - first, P::ChunkElements);
        if (!inner && generation >= kSlots) {
            if (slot == 0) CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE3>(kPanel0Free);
            else CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE3>(kPanel1Free);
        }
        wait_flag(PIPE_V, PIPE_MTE3, ::event_t(output_slot));
        const int64_t destination = (int64_t(core) * kSlots + slot) *
            (Shape::workspace_panel_bytes / kHalfBytes) + lane * P::LocalElements + first;
        asc_copy_ub2gm(workspace_ + destination,
            ub_ptr<half>(P::DecodedOffset + output_slot * P::DecodeCapacity), uint32_t(elements * kHalfBytes));
        if (inner + 1 == P::Blocks) {
            if (slot == 0) CrossCoreSetFlag<kGroupFlagMode, PIPE_MTE3>(kPanel0Ready);
            else CrossCoreSetFlag<kGroupFlagMode, PIPE_MTE3>(kPanel1Ready);
        }
        if (count < 0 || block + kSlots < count * P::Blocks)
            set_flag(PIPE_MTE3, PIPE_V, ::event_t(output_slot));
    }
  public:
    __aicore__ inline void run(GM_ADDR packed, GM_ADDR scales, GM_ADDR routing, GM_ADDR workspace,
                               int experts, int cores) {
        packed_ = packed; scales_ = scales; workspace_ = reinterpret_cast<__gm__ half *>(workspace);
        const int core = GetBlockIdx() / kVectorLanesPerCube, lane = GetSubBlockIdx();
        grouped_gemm::Schedule<P::Rows, P::Columns> schedule(routing, experts, Shape::panels, Shape::n);
        Tile tile{};
        if (!schedule.next(core, tile)) return;
        initialize();
        const int known = schedule.known_steps(core, cores);
        int count = known;
        for (int step = 0; count < 0 || step < count * P::Blocks + kSlots; ++step) {
            const int generation = step / P::Blocks;
            if (step % P::Blocks == 0 && (count < 0 || generation < count)) {
                if (generation % Shape::panels == 0 &&
                    !schedule.next(core + generation / Shape::panels * cores, tile)) count = generation;
                else load_weights(tile, generation, lane);
            }
            // Route advancement above must reveal the final count before a
            // store decides whether to publish another output-reuse credit.
            // Publish completed chunks promptly. Only a GM-slot credit wait
            // can block Scalar: in that case queue the other slot's V work
            // first, then wait immediately before overwriting GM.
            const bool credit_wait = step >= kSlots * (P::Blocks + 1) &&
                                     (step - kSlots) % P::Blocks == 0;
            if (step >= kSlots && !credit_wait) store_block(step - kSlots, count, core, lane);
            if (step >= 1 && (count < 0 || step - 1 < count * P::Blocks)) compute_block(step - 1, known);
            if (credit_wait) store_block(step - kSlots, count, core, lane);
        }
        for (int slot = 0; slot < kSlots; ++slot)
            if (known < 0 && count > slot) {
                wait_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
                wait_flag(PIPE_V, PIPE_MTE2, ::event_t(slot + kScaleEventBase));
                if (slot == 0) CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE3>(kPanel0Free);
                else CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE3>(kPanel1Free);
            }
        if (known < 0 && count > 0) {
            native_device::pipe_fence<PIPE_MTE2, PIPE_S, kRetirementEvent>();
        }
        native_device::pipe_fence<PIPE_MTE3, PIPE_S, kRetirementEvent>();
    }
};
} // namespace quant_gmm
