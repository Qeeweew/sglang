#pragma once
#include <qli/state.h>

namespace qli {
inline constexpr int32_t kNegativeInfinityBits = int32_t(0xff800000u);
inline constexpr int32_t kNegativePenaltyBits = int32_t(0xf149f2cau); // FP32 -1e30

// SDK sorting primitives allocate no private scratch or events here. This
// class owns UB spans and dependencies. ID/value extraction retains the
// SDK normal-mask and reserved-count contract. Rank pairs
// store FP32 scores and bit-preserved INT32 IDs.
class RankPipeline {
  public:
    __aicore__ static void initialize_sequence() {
        auto sequence = ub_pointer<int32_t>(UbLayout::sequence);
        {
            set_flag(PIPE_V, PIPE_S, ::event_t(kImmediateEvent));
            wait_flag(PIPE_V, PIPE_S, ::event_t(kImmediateEvent));
        }
        for (uint32_t lane = 0; lane < kDmaWords; ++lane)
            sequence[lane] = lane;
        {
            set_flag(PIPE_S, PIPE_V, ::event_t(kImmediateEvent));
            wait_flag(PIPE_S, PIPE_V, ::event_t(kImmediateEvent));
        }
#pragma unroll
        for (uint32_t span = kDmaWords; span < kKeyTileCapacity; span *= 2) {
            asc_add_scalar((sequence + (span)), sequence, int32_t(span), span);
            pipe_barrier(PIPE_V);
        }
    }
    __aicore__ static void sequence(__ubuf__ int32_t *ids, int32_t first, uint32_t count) {
        asc_add_scalar(ids, ub_pointer<int32_t>(UbLayout::sequence), first, count);
    }
    __aicore__ static constexpr uint32_t sort_count(uint32_t count) {
        uint32_t result = kSortRun;
        while (result < count)
            result *= kSortMergeInputs;
        return result;
    }
    __aicore__ static void empty(__ubuf__ float *pairs, uint32_t count) {
        uint64_t even[2]{0x5555555555555555ULL, 0};
        uint64_t odd[2]{0xaaaaaaaaaaaaaaaaULL, 0};
        {
            asc_set_vector_mask(even[1], even[0]);
            asc_duplicate(reinterpret_cast<__ubuf__ int32_t *>(pairs), kNegativeInfinityBits,
                          uint8_t(count * kRankFields / kVectorWords), 1, kVectorBlocks);
        };
        {
            asc_set_vector_mask(odd[1], odd[0]);
            asc_duplicate(reinterpret_cast<__ubuf__ int32_t *>(pairs), -1,
                          uint8_t(count * kRankFields / kVectorWords), 1, kVectorBlocks);
        };
        pipe_barrier(PIPE_V);
    }
    __aicore__ static __ubuf__ float *sort(__ubuf__ float *values, __ubuf__ uint32_t *indices,
                                           uint32_t count) {
        auto current = ub_pointer<float>(UbLayout::sort0),
             next = ub_pointer<float>(UbLayout::sort1);
        if ASCEND_IS_AIV {
            vbitsort(current, values, indices, uint8_t(count / kSortRun));
        }
        pipe_barrier(PIPE_V);
        for (uint32_t run = kSortRun; run < count; run *= kSortMergeInputs) {
            __ubuf__ float *input[kSortMergeInputs]{current, current + 2 * run, current + 4 * run,
                                                    current + 6 * run};
            // arch2201 VMS4 packs four uint16 lengths and repeat/valid-list bits.
            constexpr uint32_t kLengthBits = 16, kValidListShift = 8;
            const uint64_t lengths = uint64_t(run) | (uint64_t(run) << kLengthBits) |
                                     (uint64_t(run) << (2 * kLengthBits)) |
                                     (uint64_t(run) << (3 * kLengthBits));
            const uint64_t control = count / (kSortMergeInputs * run) |
                                     (uint64_t((1u << kSortMergeInputs) - 1) << kValidListShift);
            if ASCEND_IS_AIV {
                vmrgsort4(next, input, lengths, control);
            }
            pipe_barrier(PIPE_V);
            auto temporary = current;
            current = next;
            next = temporary;
        }
        return current;
    }
    __aicore__ static void merge(__ubuf__ float *accumulator, uint32_t keep,
                                 __ubuf__ float *incoming, uint32_t count) {
        __ubuf__ float *input[kSortMergeInputs]{accumulator, incoming, incoming, incoming};
        constexpr uint32_t kLengthBits = 16, kValidListShift = 8;
        const uint64_t lengths = uint64_t(keep) | (uint64_t(count) << kLengthBits);
        constexpr uint64_t kControl =
            1 | (uint64_t((1u << kRankMergeInputs) - 1) << kValidListShift);
        auto result = ub_pointer<float>(UbLayout::merge);
        if ASCEND_IS_AIV {
            vmrgsort4(result, input, lengths, kControl);
        }
        pipe_barrier(PIPE_V);
        // Integer V copy preserves index bits, including -1's NaN bit pattern.
        asc_add_scalar(reinterpret_cast<__ubuf__ int32_t *>(accumulator),
                       reinterpret_cast<__ubuf__ int32_t *>(result), 0, keep * 2);
        pipe_barrier(PIPE_V);
    }
    __aicore__ static void extract_ids(__ubuf__ int32_t *destination, __ubuf__ float *pairs,
                                       uint32_t count) {
        if ASCEND_IS_AIV {
            asc_set_mask_norm();
            asc_set_vector_mask(0, 0);
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(destination),
                      reinterpret_cast<__ubuf__ uint32_t *>(pairs),
                      reinterpret_cast<__ubuf__ uint32_t *>(pairs) +
                          kVectorBytes * sizeof(uint32_t),
                      uint16_t((count * kRankFields + kVectorWords - 1) / kVectorWords), 1,
                      uint8_t(2), kVectorBlocks, 0);
            (void)asc_get_rsvd_count();
            asc_set_mask_norm();
        };
        pipe_barrier(PIPE_V);
    }
    __aicore__ static void extract_values(__ubuf__ float *destination, __ubuf__ float *pairs,
                                          uint32_t count) {
        if ASCEND_IS_AIV {
            asc_set_mask_norm();
            asc_set_vector_mask(0, 0);
            vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(destination),
                      reinterpret_cast<__ubuf__ uint32_t *>(pairs),
                      reinterpret_cast<__ubuf__ uint32_t *>(pairs) +
                          kVectorBytes * sizeof(uint32_t),
                      uint16_t((count * kRankFields + kVectorWords - 1) / kVectorWords), 1,
                      uint8_t(1), kVectorBlocks, 0);
            (void)asc_get_rsvd_count();
            asc_set_mask_norm();
        };
        pipe_barrier(PIPE_V);
    }
    __aicore__ static void mask_invalid(__ubuf__ float *scores, __ubuf__ int32_t *ids,
                                        uint32_t base, uint32_t valid, uint32_t count) {
        sequence(ids, int32_t(base), count);
        pipe_barrier(PIPE_V);
        const uint32_t active = valid > base ? min(valid - base, count) : 0;
        if (active == count)
            return;
        uint32_t first = active / kVectorWords * kVectorWords;
        const uint32_t lane = active % kVectorWords;
        if (lane) {
            const uint32_t lanes = min(kVectorWords, count - first);
            const uint64_t high = lanes == kVectorWords ? ~uint64_t(0) : (uint64_t(1) << lanes) - 1;
            uint64_t suffix[2]{high & (~uint64_t(0) << lane), 0};
            // sequence's count API restores normal mode. Both destinations
            // start on a full repeat boundary; count32 never writes lane32+.
            {
                asc_set_vector_mask(suffix[1], suffix[0]);
                asc_duplicate((reinterpret_cast<__ubuf__ int32_t *>(scores) + (first)),
                              kNegativeInfinityBits, 1, 1, kVectorBlocks);
            };
            {
                asc_set_vector_mask(suffix[1], suffix[0]);
                asc_duplicate((ids + (first)), -1, 1, 1, kVectorBlocks);
            };
            pipe_barrier(PIPE_V);
            first += lanes;
        }
        if (first < count) {
            asc_duplicate((reinterpret_cast<__ubuf__ int32_t *>(scores) + (first)),
                          kNegativeInfinityBits, count - first);
            asc_duplicate((ids + (first)), -1, count - first);
            pipe_barrier(PIPE_V);
        }
    }
};
} // namespace qli
