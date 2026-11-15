#pragma once
#include <cstdint>
namespace dspark_schedule {
// arch2201 ONE_BLK_SIZE: each worker owns complete FP32 output DMA blocks.
inline constexpr uint32_t kDataBlockBytes = 32;
inline constexpr uint32_t kRequestGroup = kDataBlockBytes / sizeof(float);
} // namespace dspark_schedule
extern "C" void dsv41_dspark_schedule_prepare_launch(void *stream, uint32_t blocks,
                                                     uint8_t *confidence, uint8_t *survival,
                                                     int64_t rows, int64_t gamma, int64_t columns,
                                                     int64_t stride, float eps);
extern "C" void dsv41_dspark_schedule_finish_launch(void *stream, uint32_t blocks,
                                                    uint8_t *survival, uint8_t *output,
                                                    int64_t rows, int64_t columns, int64_t stride,
                                                    int64_t budget, int64_t minimum,
                                                    int64_t maximum);
