#pragma once
#include <common/device.h>
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <qli/layout.h>

namespace qli {
using native_device::ub_ptr;
using namespace AscendC;
inline constexpr uint32_t kImmediateEvent = 0;

inline constexpr int kHeads = QLI_HEADS;
inline constexpr int kDepth = QLI_DEPTH;
inline constexpr int kQueries = QLI_QUERY_ROWS;
inline constexpr int kTopk = QLI_TOPK;
inline constexpr int kMode = QLI_MODE;
inline constexpr uint32_t kWorkers = QLI_WORKERS;
inline constexpr uint32_t kScoreWorkers = QLI_SCORE_WORKERS;
inline constexpr uint32_t kPageRows = QLI_PAGE_ROWS;
inline constexpr uint32_t kPageColumns = QLI_PAGE_COLUMNS;
inline constexpr uint32_t kKeyPageStride = QLI_KEY_PAGE_STRIDE;
inline constexpr uint32_t kScalePageStride = QLI_SCALE_PAGE_STRIDE;
inline constexpr uint32_t kRatio = QLI_RATIO;
inline constexpr uint32_t kKeySplits = QLI_KEY_SPLITS;
inline constexpr int kMatrixRows = kHeads * kQueries;
inline constexpr uint32_t kQueryOperandBytes = kMatrixRows * kDepth * sizeof(int8_t);
inline constexpr uint32_t kWeightOperandBytes = kCubeRows * kMatrixRows * sizeof(half);
// L0A can retain both task-invariant left operands if their actual extents fit.
// The largest legal Q has a full L0A and retains the streamed import schedule.
inline constexpr bool kCachedLeftOperands =
    kQueryOperandBytes + kWeightOperandBytes <= arch2201::kL0ABytes;
inline constexpr uint32_t kWeightOperandOffset = kCachedLeftOperands ? kQueryOperandBytes : 0;
inline constexpr uint32_t kInt8KeyLimit = arch2201::kL0BBytes / (kDepth * sizeof(int8_t));
inline constexpr uint32_t kQkKeyLimit =
    arch2201::kL0CBytes / kC0Partitions / (kMatrixRows * sizeof(int32_t));
inline constexpr uint32_t kWeightedKeyLimit = arch2201::kL0BBytes / (kMatrixRows * sizeof(half));
inline constexpr uint32_t kExportKeyLimit =
    arch2201::kL0CBytes / kC0Partitions / (kCubeRows * sizeof(float));
inline constexpr uint32_t kOperandKeyLimit =
    kInt8KeyLimit < kWeightedKeyLimit ? kInt8KeyLimit : kWeightedKeyLimit;
inline constexpr uint32_t kResultKeyLimit =
    kQkKeyLimit < kExportKeyLimit ? kQkKeyLimit : kExportKeyLimit;
inline constexpr uint32_t kBudgetKeyLimit =
    kOperandKeyLimit < kResultKeyLimit ? kOperandKeyLimit : kResultKeyLimit;
// Independent right operands permit QK(i) to precede head(i-1). Their combined
// L0B span, rather than either view in isolation, bounds this compile-time N.
inline constexpr bool kPipelinedRightOperands = kCachedLeftOperands;
inline constexpr uint32_t kCombinedRightKeyLimit =
    arch2201::kL0BBytes / (kDepth * sizeof(int8_t) + kMatrixRows * sizeof(half));
inline constexpr uint32_t kPipelineKeyLimit =
    kPipelinedRightOperands && kCombinedRightKeyLimit < kBudgetKeyLimit ? kCombinedRightKeyLimit
                                                                        : kBudgetKeyLimit;
inline constexpr uint32_t kKeyRows =
    (kPipelineKeyLimit < kKeyTileCapacity ? kPipelineKeyLimit : kKeyTileCapacity) / kCubeRows *
    kCubeRows;
static_assert(kKeyRows >= kCubeRows);
inline constexpr uint32_t kHeadOperandOffset =
    kPipelinedRightOperands ? kKeyRows * kDepth * sizeof(int8_t) : 0;
static_assert(!kPipelinedRightOperands ||
              kHeadOperandOffset + kKeyRows * kMatrixRows * sizeof(half) <= arch2201::kL0BBytes);
inline constexpr uint64_t kScoreCoreBytes =
    kScoreSlots * kQueries * kKeyTileCapacity * sizeof(float);
inline constexpr uint64_t kWeightCoreBytes = kCubeRows * kMatrixRows * sizeof(half);
inline constexpr uint64_t kRankListWords =
    kRankFields * (align32(kTopk) + (kMode == kSourceCandidates ? kKeyTileCapacity : 0));
inline constexpr int kKeep = align32(kTopk);
inline constexpr int kLocalQueries = (kQueries + kVectorActors - 1) / kVectorActors;
static_assert(kMatrixRows <= int(kQueryHeadCapacity));

// Private arenas use no TPipe, implicit scratch or framework-owned events.
struct UbLayout {
    static constexpr uint32_t scores = 0;
    static constexpr uint32_t half_scales = scores + kKeyTileCapacity * sizeof(float);
    static constexpr uint32_t float_scales = half_scales + kKeyTileCapacity * sizeof(half);
    static constexpr uint32_t indices = float_scales + kKeyTileCapacity * sizeof(float);
    static constexpr uint32_t sort0 = indices + kKeyTileCapacity * sizeof(int32_t);
    static constexpr uint32_t sort1 = sort0 + 2 * kKeyTileCapacity * sizeof(float);
    static constexpr uint32_t merge = sort1 + 2 * kKeyTileCapacity * sizeof(float);
    static constexpr uint32_t tokens = merge + 4 * kKeyTileCapacity * sizeof(float);
    static constexpr uint32_t mode = tokens + kLocalQueries * 2 * kKeep * sizeof(float);
    static constexpr uint32_t aux =
        mode + (kMode == kSourceCandidates ? kLocalQueries * 2 * kKeyTileCapacity * sizeof(float)
                : kMode == kConsumerCandidates ? kLocalQueries * kKeyTileCapacity * sizeof(float)
                                               : 0);
    // candidate_penalty owns two full INT32 tile spans. Weight packing and raw
    // candidate input reuse that span at distinct stages of the vector pipe.
    static constexpr uint32_t aux_bytes = 2 * kKeyTileCapacity * sizeof(int32_t);
    static constexpr uint32_t sequence = aux + aux_bytes;
    static constexpr uint32_t bytes = sequence + kKeyTileCapacity * sizeof(int32_t);
};
static_assert(UbLayout::bytes <= arch2201::kUbBytes);
static_assert((kCubeRows + 2) * kMatrixRows * sizeof(half) <= UbLayout::aux_bytes);
static_assert(kKeyTileCapacity * sizeof(int32_t) <= UbLayout::aux_bytes);
struct L1Layout {
    static constexpr uint32_t query = 0;
    static constexpr uint32_t weights = query + kMatrixRows * kDepth;
    static constexpr uint32_t key = weights + kCubeRows * kMatrixRows * sizeof(half);
    static constexpr uint32_t relu = key + kKeySlots * kKeyRows * kDepth;
    static constexpr uint32_t bytes = relu + kReluSlots * kMatrixRows * kKeyRows * sizeof(half);
};
static_assert(L1Layout::bytes <= arch2201::kL1Bytes);
static_assert(kMatrixRows * kKeyRows * sizeof(int32_t) <= arch2201::kL0CBytes / 2);

template <class T> __aicore__ inline __gm__ T *gm_pointer(uintptr_t address) {
    return reinterpret_cast<__gm__ T *>(address);
}
template <class T> __aicore__ inline __cbuf__ T *l1_pointer(uint32_t offset) {
    return reinterpret_cast<__cbuf__ T *>(uint64_t(offset));
}
template <class T> __aicore__ inline __ca__ T *l0a_pointer(uint32_t offset = 0) {
    return reinterpret_cast<__ca__ T *>(uint64_t(offset));
}
template <class T> __aicore__ inline __cb__ T *l0b_pointer(uint32_t offset = 0) {
    return reinterpret_cast<__cb__ T *>(uint64_t(offset));
}
template <class T> __aicore__ inline __cc__ T *l0c_pointer(uint32_t offset = 0) {
    return reinterpret_cast<__cc__ T *>(uint64_t(offset));
}
// Flags belong to one AIC and its two AIVs. Both vectors participate even if
// a query tail gives one vector no rows. No flag survives invocation return.
enum CrossFlag : uint16_t {
    WeightReady = 0,
    ScoreReady0 = 1,
    ScoreReady1 = 2,
    ScoreCredit0 = 3,
    ScoreCredit1 = 4,
    WeightCredit = 5
};
struct Workspace {
    // The named entry argument lives until the final PIPE_ALL drain. Views
    // share that packet instead of copying all tensor pointers/configuration.
    const Sources &args;
    uint32_t core;
    __aicore__ __gm__ float *scores(int slot) const {
        return gm_pointer<float>(args.workspace) +
               ((uint64_t(core) * kScoreSlots + slot) * kQueries * kKeyTileCapacity);
    }
    __aicore__ __gm__ half *weights() const {
        return gm_pointer<half>(args.workspace + kScoreWorkers * kScoreCoreBytes) +
               (uint64_t(core) * kCubeRows * kMatrixRows);
    }
    __aicore__ __gm__ float *ranks(uint32_t row, uint32_t split) const {
        return gm_pointer<float>(args.workspace + kScoreWorkers * kScoreCoreBytes +
                                 kScoreWorkers * kWeightCoreBytes) +
               ((uint64_t(row) * kKeySplits + split) * kRankListWords);
    }
};
} // namespace qli
