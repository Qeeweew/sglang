#pragma once
#include <climits>
#include <cstdint>

namespace fp8_weight {
// Checkpoint E4M3FN values share one E8M0 exponent per 32 x 32 tile.
inline constexpr uint32_t kScaleGroup = 32;
// Four neighboring scale columns amortize Gather setup in the same pipeline.
inline constexpr uint32_t kTileScaleColumns = 4;
inline constexpr uint32_t kTileColumns = kTileScaleColumns * kScaleGroup;
inline constexpr uint32_t kTileRows = kScaleGroup;
inline constexpr uint32_t kTileElements = kTileRows * kTileColumns;
inline constexpr uint32_t kAlphabet = 1U << (CHAR_BIT * sizeof(uint8_t));
inline constexpr uint32_t kLutElements = 2 * kAlphabet; // values, then exponents
} // namespace fp8_weight
