#pragma once
#include <mhc/params.h>
namespace mhc {
inline constexpr uint32_t kBranches = MHC_BRANCHES, kWidth = MHC_WIDTH;
inline constexpr uint32_t kWorkers = MHC_WORKERS;
inline constexpr uint32_t kInputWidth = kBranches * kWidth;
inline constexpr uint32_t kProjectionColumns = kBranches * (kBranches + 2);
inline constexpr uint32_t kProjectionStride = round_up(kProjectionColumns, kMatrixRows);
#if defined(MHC_PROJECT_PLAN)
inline constexpr uint32_t kTileRows = MHC_TILE_ROWS, kPanelK = MHC_PANEL_K;
inline constexpr uint32_t kSplits = MHC_SPLITS, kRowGroups = MHC_ROW_GROUPS;
inline constexpr uint32_t kPanelCount = (kInputWidth + kPanelK - 1) / kPanelK;
inline constexpr ProjectParams kProjectPlan{0,
                                            0,
                                            kWidth,
                                            kBranches,
                                            kInputWidth,
                                            kProjectionColumns,
                                            kProjectionStride,
                                            kTileRows,
                                            kPanelK,
                                            kPanelCount,
                                            MHC_L1_TILE_K,
                                            kSplits,
                                            kRowGroups,
                                            kSplits * kRowGroups,
                                            0, // Host-only Vector budget is not read by the device.
                                            0};
#elif defined(MHC_STATS_PLAN)
inline constexpr uint32_t kStatsRows = MHC_TILE_ROWS;
inline constexpr uint32_t kPostTile = MHC_COLUMN_TILE;
inline constexpr StatsParams kStatsPlan{0,
                                        0,
                                        kBranches,
                                        kWidth,
                                        kProjectionColumns,
                                        kProjectionStride,
                                        MHC_SPLITS,
                                        kStatsRows,
                                        kPostTile,
                                        kPostTile ? (kWidth + kPostTile - 1) / kPostTile : 0,
                                        1.0f / kInputWidth,
                                        0};
#else
inline constexpr uint32_t kColumnTile = MHC_COLUMN_TILE;
inline constexpr uint32_t kColumnTiles = (kWidth + kColumnTile - 1) / kColumnTile;
inline constexpr uint32_t kOutputColumns = MHC_OUTPUT_COLUMNS;
inline constexpr StateParams kStatePlan{0,        kWidth, kBranches,     kColumnTile,
                                        kColumnTiles, 0,      kOutputColumns};
inline constexpr float kInverseWidth = 1.0f / kWidth;
#endif
} // namespace mhc
