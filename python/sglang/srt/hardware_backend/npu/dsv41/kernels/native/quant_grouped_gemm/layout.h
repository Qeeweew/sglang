#pragma once
#include <kernel_operator.h>
#include <quant_grouped_gemm/tiling.h>

namespace quant_gmm {
inline constexpr int kHalfBytes = sizeof(half), kSlots = 2, kPackedPerByte = 2;
inline constexpr int kScaleGroup = 32, kFractalRows = 16, kDmaBytes = 32, kVectorBytes = 256;
inline constexpr int kHalfVectorElements = kVectorBytes / kHalfBytes;
inline constexpr int16_t kRankCorrectionBits = 0x4400;
inline constexpr float kTaggedScaleCorrection = -0x1p-10f;
inline constexpr int kAccumulatorContinue = 2, kAccumulatorPublish = 3;
inline constexpr int kOperandReuseEvent = 3, kFootprintRetiredEvent = 5;
inline constexpr int kMatrixStateEvent = 6, kFixStateEvent = 0, kScaleEventBase = 2;
inline constexpr int kRightOperandEventBase = 2, kRetirementEvent = 7, kGroupFlagMode = 2;
enum PanelFlag : uint16_t { kPanel0Ready = 0, kPanel1Ready = 1, kPanel0Free = 2, kPanel1Free = 3 };
constexpr int dma_align(int bytes) { return (bytes + kDmaBytes - 1) / kDmaBytes * kDmaBytes; }

template <int TileRows, int TileColumns> struct Layout {
    static constexpr int Rows = TileRows, Columns = TileColumns;
    static constexpr Storage Memory{QUANT_GMM_K, Rows, Columns};
    static constexpr int C0Elements = Storage::kC0Elements;
    static constexpr int AtomElements = Storage::kAtomElements;
    static constexpr int LocalBlocks = Columns / (kVectorLanesPerCube * C0Elements);
    static constexpr int StepK = Memory.step_k, DotK = Memory.cube_k;
    static constexpr int LocalElements = StepK * (Columns / kVectorLanesPerCube);
    static constexpr int CodeCapacity = Memory.local_elements / kPackedPerByte;
    static constexpr int ScaleSlotBytes = Memory.local_elements / kScaleGroup * kHalfBytes;
    static constexpr int ScaleElementBytes = sizeof(uint8_t);
    static constexpr int ScaleGroups = StepK / kScaleGroup;
    static constexpr int ScaleBlockBytes = dma_align(ScaleGroups * C0Elements * ScaleElementBytes);
    static constexpr int FactorBytes = Memory.local_elements / kScaleGroup * kHalfBytes;
    static constexpr int OutputElements = Memory.output_elements;
    static constexpr int DecodeCapacity = OutputElements * kHalfBytes;
    static constexpr int MicroElements = Memory.micro_elements;
    // Capacity is a UB budget; balance valid data over both output slots
    // so the first MTE3 chunk can overlap decoding the second one.
    static constexpr int Blocks = kSlots;
    static constexpr int ChunkElements =
        (LocalElements / Blocks + AtomElements - 1) / AtomElements * AtomElements;
    static constexpr int ScaleOffset = kSlots * CodeCapacity;
    static constexpr int ScaleOutputOffset = ScaleOffset + kSlots * ScaleSlotBytes;
    static constexpr int ThresholdOffset = ScaleOutputOffset + FactorBytes;
    // The second span keeps the absolute scratch on the preceding scale
    // span's256-byte phase; decoded/absolute add sources use opposite phases.
    static constexpr int AbsoluteOffset = ThresholdOffset + 2 * kVectorBytes;
    static constexpr int DecodedOffset = AbsoluteOffset + MicroElements * kHalfBytes + kVectorBytes;
    static constexpr int UbBytes = DecodedOffset + kSlots * DecodeCapacity;
    static constexpr int A1Bytes = Rows * StepK * kHalfBytes;
    static constexpr int B1Bytes = Columns * StepK * kHalfBytes;
    static constexpr int A0Bytes = Rows * DotK * kHalfBytes;
    static constexpr int B0Bytes = Columns * DotK * kHalfBytes;
    static_assert(Blocks >= 1 && Blocks <= 2);
    static_assert(ChunkElements <= OutputElements && ChunkElements < LocalElements);
    static_assert(LocalBlocks * ScaleBlockBytes <= ScaleSlotBytes);
    static_assert(OutputElements % AtomElements == 0 && MicroElements % AtomElements == 0);
    static_assert(MicroElements / kHalfVectorElements <= 255 && UbBytes <= Storage::kUbArenaBytes);
    static_assert(kSlots * (A1Bytes + B1Bytes) <= arch2201::kL1Bytes);
    static_assert(kSlots * A0Bytes <= arch2201::kL0ABytes && kSlots * B0Bytes <= arch2201::kL0BBytes);
    static_assert(Rows * Columns * sizeof(float) <= arch2201::kL0CBytes);
};
template <class Plan> struct PanelShape {
    static constexpr int k = QUANT_GMM_K, n = QUANT_GMM_N;
    static constexpr int step_k = Plan::StepK, panels = k / step_k;
    static constexpr int chunks = (step_k + Plan::DotK - 1) / Plan::DotK;
    static constexpr int workspace_panel_bytes = step_k * Plan::Columns * sizeof(half);
    __aicore__ inline static int matrix_rows(int valid) {
        if constexpr (Plan::Rows == kFractalRows) return kFractalRows;
        return (valid + kFractalRows - 1) / kFractalRows * kFractalRows;
    }
};
} // namespace quant_gmm
