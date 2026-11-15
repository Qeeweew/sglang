#pragma once
#include <cstdint>
extern "C" void dsv41_engram_hash_launch(void *stream, uint32_t blocks, uint8_t *ids, uint8_t *pos,
                                         uint8_t *cu, uint8_t *slots, uint8_t *history,
                                         uint8_t *active, uint8_t *map, uint8_t *mult,
                                         uint8_t *primes, uint8_t *offsets, uint8_t *out,
                                         int64_t rows, int64_t batch, int64_t history_rows,
                                         int64_t vocab, int64_t pad, int64_t image,
                                         int64_t mm_shift, uint32_t layers, uint32_t history_tokens,
                                         uint32_t heads, uint32_t hash_columns);
extern "C" void dsv41_engram_commit_launch(void *stream, uint32_t blocks, uint8_t *ids, uint8_t *cu,
                                           uint8_t *slots, uint8_t *accepted, uint8_t *history,
                                           int64_t rows, int64_t batch, int64_t history_rows,
                                           uint32_t history_tokens);
extern "C" void dsv41_engram_gate_launch(void *stream, uint32_t blocks, uint8_t *hidden,
                                         uint8_t *kv, uint8_t *qw, uint8_t *kw, uint8_t *active,
                                         uint8_t *output, int64_t tokens, int64_t D,
                                         int64_t branches, float inv_sqrt_hidden);
