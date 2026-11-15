#pragma once
#include <cstdint>
extern "C" void dsv41_compressor_prepare_launch(void *stream, uint32_t blocks, uint8_t *kv,
                                                uint8_t *scores, uint8_t *gamma, uint8_t *cu,
                                                uint8_t *requests, uint8_t *pending,
                                                uint8_t *lengths, uint8_t *pooled, uint8_t *latent,
                                                uint8_t *positions, uint8_t *rope_positions,
                                                uint8_t *starts, int64_t tokens, int64_t batches,
                                                int64_t slots, int64_t ratio, int64_t width);
extern "C" void dsv41_compressor_commit_launch(void *stream, uint32_t blocks, uint8_t *kv,
                                               uint8_t *scores, uint8_t *cu, uint8_t *requests,
                                               uint8_t *starts, uint8_t *accepted, uint8_t *pending,
                                               uint8_t *lengths, uint8_t *positions,
                                               uint8_t *rope_positions, uint8_t *status,
                                               int64_t batches, int64_t slots, int64_t ratio,
                                               int64_t width);
