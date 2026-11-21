#pragma once
#include <kernel_operator.h>
#include <attention/launch.h>

namespace attention {
using namespace AscendC;
inline constexpr uint32_t kImmediateEvent = 0;
inline constexpr uint32_t kQueryReuseEvent = 3;

using BF16 = bfloat16_t;
inline constexpr int kMode = ATTENTION_MODE;
inline constexpr int32_t kHeads = ATTENTION_HEADS,
                         kWidth = ATTENTION_WIDTH;
inline constexpr int32_t kPageColumns = ATTENTION_PAGE_COLUMNS;
inline constexpr int32_t kCompressedColumns = ATTENTION_COMPRESSED_COLUMNS;
inline constexpr int32_t kPageRows = ATTENTION_PAGE_ROWS, kWindowRows = ATTENTION_WINDOW_ROWS;
inline constexpr int32_t kSelectedRows = ATTENTION_SELECTED_ROWS, kRatio = ATTENTION_RATIO;
inline constexpr int32_t kWorkerBudget = ATTENTION_WORKER_BUDGET;
inline constexpr int32_t kVerifyRingCapacity = ATTENTION_VERIFY_RING_CAPACITY;
inline constexpr bool kWindowRing = ATTENTION_WINDOW_RING;
inline constexpr bool kCoordinate32 = ATTENTION_COORDINATE32,
                      kCoordinateElements32 = ATTENTION_COORDINATE_ELEMENTS32;
inline constexpr int32_t kCompressedPageRows = kPageRows / kRatio;
__aicore__ constexpr int32_t page_log2(int32_t rows) {
    int32_t shift = 0;
    while ((int64_t(1) << shift) < rows)
        ++shift;
    return (int64_t(1) << shift) == rows ? shift : -1;
}
inline constexpr int32_t kCoordinateShift = page_log2(kCompressedPageRows);
inline constexpr int32_t kPageShift = kCoordinateShift < 0 ? -1 : kCoordinateShift + kRatio - 1;
inline constexpr int32_t kCapacity = kMode == 0
                                         ? ((kWindowRows + kDotRows - 1) / kDotRows) * kDotRows
                                     : kMode == 1 ? kSparseWindowCapacity + kSelectedRows
                                                  : ATTENTION_DRAFT_CAPACITY;
inline constexpr int32_t kTileRows = ATTENTION_TILE_ROWS;
inline constexpr float kScale = ATTENTION_SCALE;
inline constexpr bool kSparse = kMode == int(Mode::Sparse);
inline constexpr bool kDraft = kMode == int(Mode::Draft);
inline constexpr uint32_t kDmaBytes = 32;
// The entry declares MIX_AIC_1_2; this is a task-format ratio, not core count.
inline constexpr int32_t kVectorActorsPerCube = 2;
inline constexpr uint32_t kVectorElements = 256 / sizeof(float);
inline constexpr uint32_t kBlockElements = kDmaBytes / sizeof(float);
inline constexpr uint32_t kGatherRows = 32;
inline constexpr uint32_t kSparseHeadTile = 16;
inline constexpr uint32_t kPageTableCapacity = 1024;
inline constexpr uint32_t kGatherBytes = 2 * kGatherRows * kWidthCapacity * sizeof(BF16);
inline constexpr uint32_t kScoreBytes = 2 * kSparseHeadTile * kSparseRowCapacity * sizeof(float);
inline constexpr uint32_t kProbabilityBytes = kSparseHeadTile * kSparseRowCapacity * sizeof(BF16);
inline constexpr uint32_t kLocalHeadCapacity = kHeadCapacity / 2;
inline constexpr uint32_t kHeadStatisticsBytes = kLocalHeadCapacity * sizeof(float);
inline constexpr uint32_t kBroadcastBytes = kLocalHeadCapacity * kBlockElements * sizeof(float);
inline constexpr uint32_t kMaskSlotBytes = kGatherRows * sizeof(float);

// UB arenas have explicit ownership. Gather[2] is MTE2/Scalar -> MTE3;
// Score planes[2] are MTE2 -> Vector; P is Vector -> MTE3. During finish,
// P and its following bias/partial storage become the output arena. Statistics and lookup
// tables remain disjoint until invocation exit.
struct UbLayout {
    static constexpr uint32_t gather = 0;
    static constexpr uint32_t scores = kSparse ? kGatherBytes : kGatherBytes / 2;
    static constexpr uint32_t probabilities = scores + kScoreBytes;
    static constexpr uint32_t bias =
        probabilities + (kSparse ? kProbabilityBytes : 2 * kProbabilityBytes);
    static constexpr uint32_t partial =
        bias + (kSparse ? kSparseRowCapacity : kSequenceCapacity) * sizeof(float);
    static constexpr uint32_t statistics =
        partial + (kSparse
                       ? 2 * kSparseHeadTile * kSparseRowCapacity / kBlockElements * sizeof(float)
                       : 2 * 4 * kSequenceCapacity / kBlockElements * sizeof(float));
    static constexpr uint32_t sink = statistics;
    static constexpr uint32_t maximum = sink + kHeadStatisticsBytes;
    static constexpr uint32_t sum = maximum + kHeadStatisticsBytes;
    static constexpr uint32_t reduction_merge = sum + kHeadStatisticsBytes;
    static constexpr uint32_t reciprocal_refinement = reduction_merge + kHeadStatisticsBytes;
    static constexpr uint32_t broadcast = reciprocal_refinement + kHeadStatisticsBytes;
    static constexpr uint32_t final_denominator = broadcast + kBroadcastBytes;
    static constexpr uint32_t indices = final_denominator + 2 * kHeadStatisticsBytes;
    static constexpr uint32_t masks = indices + kSelectedCapacity * sizeof(int32_t);
    static constexpr uint32_t page_table = masks + 2 * kMaskSlotBytes;
    static constexpr uint32_t row_indices = page_table + kPageTableCapacity * sizeof(int32_t);
    static constexpr uint32_t offsets = row_indices + kSelectedCapacity * sizeof(int32_t);
    static constexpr uint32_t sinks = offsets + kPageTableCapacity * sizeof(int32_t);
    static constexpr uint32_t bytes =
        kSparse ? sinks + kHeadStatisticsBytes : masks + kMaskSlotBytes;
};
static_assert(UbLayout::bytes <= arch2201::kUbBytes);
inline constexpr uint32_t kL1CapacityBytes = kHeadCapacity * kWidthCapacity * sizeof(BF16) +
                                             2 * kDotRows * kFeaturePanelColumns * sizeof(BF16) +
                                             2 * kDotRows * kWidthCapacity * sizeof(BF16) +
                                             2 * kHeadCapacity * kDotRows * sizeof(BF16);
static_assert(kL1CapacityBytes <= arch2201::kL1Bytes);
inline constexpr uint32_t kL0ABytes = kHeadCapacity * kWidthCapacity * sizeof(BF16);
inline constexpr uint32_t kL0BBytes = 2 * kDotRows * kDotRows * sizeof(BF16);
inline constexpr uint32_t kL0CBytes = 2 * kHeadCapacity * kDotRows * sizeof(float);
static_assert(kL0ABytes <= arch2201::kL0ABytes && kL0BBytes <= arch2201::kL0BBytes &&
              kL0CBytes <= arch2201::kL0CBytes);

__aicore__ __attribute__((always_inline)) inline int32_t cube_actor_index() {
    if ASCEND_IS_AIV {
        return GetBlockIdx() / kVectorActorsPerCube;
    }
    return GetBlockIdx();
}
template <class T> __aicore__ inline __gm__ T *gm_pointer(uintptr_t address) {
    return reinterpret_cast<__gm__ T *>(address);
}

__aicore__ __attribute__((always_inline)) inline uint64_t valid_interval(int32_t first,
                                                                         int32_t end) {
    first = max(0, first);
    end = min(64, end);
    if (first >= end)
        return 0;
    const uint64_t upper = end == 64 ? ~uint64_t(0) : (uint64_t(1) << end) - 1;
    const uint64_t lower = first ? (uint64_t(1) << first) - 1 : 0;
    return upper & ~lower;
}

struct RequestMetadata {
    int32_t batch, count, offset, right, visible, first;
};
class RequestCursor {
    int32_t batch_ = 0, begin_ = 0, end_ = -1, length_ = 0;

  public:
    template <bool ReadWindow = true>
    __aicore__ __attribute__((always_inline)) inline RequestMetadata locate(const Sources &args,
                                                                            int32_t token) {
        const auto &s = args;
        auto boundaries = gm_pointer<int32_t>(s.cu);
        if (token < begin_ || token >= end_) {
            int32_t low = 0, high = s.batch;
            while (high - low > 1) {
                const int32_t mid = (low + high) / 2;
                if (boundaries[mid] <= token)
                    low = mid;
                else
                    high = mid;
            }
            batch_ = low;
            begin_ = boundaries[low];
            end_ = boundaries[low + 1];
            length_ = gm_pointer<int32_t>(s.lengths)[low];
        }
        const int32_t count = end_ - begin_, offset = token - begin_;
        int32_t right = max(0, length_ - count + offset + 1);
        int32_t visible = min(int32_t(kWindowRows), right), first = right - visible;
        if constexpr (kDraft) {
            first = max(0, length_ - count - kWindowRows);
            right = length_;
            visible = max(0, right - first);
            if (token < begin_ || token >= end_ || length_ < count || visible > kCapacity ||
                length_ > int64_t(kPageColumns) * kPageRows)
                visible = 0;
        } else if constexpr (ReadWindow) {
            const int32_t requested = gm_pointer<int32_t>(s.window)[token];
            if (requested > 0)
                visible = min(visible, requested);
            first = right - visible;
        }
        if constexpr (kVerifyRingCapacity)
            if (gm_pointer<int64_t>(s.verify_requests)[batch_] == 0)
                visible = 0;
        return {batch_, count, offset, right, visible, first};
    }
};

// GM ring ownership follows four cross-core flags; each includes both AIVs:
// GatherReady0: AIV MTE3 -> AIC MTE2; ScoreReady1: AIC FIX -> AIV MTE2;
// ProbabilityReady2: AIV MTE3 -> AIC MTE2; ProductReady3: AIC FIX -> AIV MTE2.
// Each actor reconstructs cursor/cache state on every invocation/Graph replay.
enum CrossCoreCredit : uint16_t {
    GatherReady = 0,
    ScoreReady = 1,
    ProbabilityReady = 2,
    ProductReady = 3
};
class Workspace {
    uintptr_t scores_, probabilities_, products_, gathered_, bias_;
    static constexpr int64_t score_elements_ = int64_t(kHeads) * kCapacity;
    static constexpr int64_t product_elements_ = int64_t(kHeads) * kWidth;
    static constexpr int64_t gather_elements_ = int64_t(kSelectedRows) * kWidth;
    static constexpr int64_t bias_elements_ = kSelectedRows;

  public:
    __aicore__
        __attribute__((always_inline)) explicit Workspace(const Sources &args, int32_t actor) {
        const int64_t score_bytes = int64_t(kWorkerBudget) * 2 * score_elements_ * sizeof(float);
        const int64_t probability_bytes = score_bytes / 2;
        const int64_t product_bytes =
            kSparse ? int64_t(kWorkerBudget) * 2 * product_elements_ * sizeof(float) : 0;
        const int64_t gather_bytes =
            kSparse ? int64_t(kWorkerBudget) * 3 * gather_elements_ * sizeof(BF16) : 0;
        const uintptr_t base = args.workspace;
        // Actor bases and ring strides are invocation-invariant. Compute them
        // once rather than rebuilding the complete GM layout for each tile.
        scores_ = base + int64_t(actor) * 2 * score_elements_ * sizeof(float);
        probabilities_ = base + score_bytes + int64_t(actor) * 2 * score_elements_ * sizeof(BF16);
        products_ = base + score_bytes + probability_bytes +
                    int64_t(actor) * 2 * product_elements_ * sizeof(float);
        gathered_ = base + score_bytes + probability_bytes + product_bytes +
                    int64_t(actor) * 3 * gather_elements_ * sizeof(BF16);
        bias_ = base + score_bytes + probability_bytes + product_bytes + gather_bytes +
                int64_t(actor) * 3 * bias_elements_ * sizeof(float);
    }
    __aicore__ __attribute__((always_inline)) __gm__ float *scores(int32_t task) const {
        return gm_pointer<float>(scores_) + ((task % 2) * score_elements_);
    }
    __aicore__ __attribute__((always_inline)) __gm__ BF16 *probabilities(int32_t task) const {
        return gm_pointer<BF16>(probabilities_) + ((task % 2) * score_elements_);
    }
    __aicore__ __attribute__((always_inline)) __gm__ float *products(int32_t task) const {
        return gm_pointer<float>(products_) + ((task % 2) * product_elements_);
    }
    __aicore__ __attribute__((always_inline)) __gm__ BF16 *gathered(int32_t task) const {
        return gm_pointer<BF16>(gathered_) + ((task % 3) * gather_elements_);
    }
    __aicore__ __attribute__((always_inline)) __gm__ float *bias(int32_t task) const {
        return gm_pointer<float>(bias_) + ((task % 3) * bias_elements_);
    }
};
} // namespace attention
