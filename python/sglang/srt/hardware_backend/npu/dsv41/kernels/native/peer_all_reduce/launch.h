#pragma once
#include <cstdint>
void prepare_peer_all_reduce();
void launch_peer_all_reduce(void *stream, void *input, void *output, void *local, void *peer,
                            void *table, uint32_t elements);
void launch_peer_group_gather(void *stream, void *in_a, void *in_b, void *out_a, void *out_b,
                              void *local, void *peer, uint32_t rank, uint32_t outer,
                              uint32_t local_rows, uint32_t valid_rows, uint32_t row_bytes_a,
                              uint32_t row_bytes_b, bool skip_own_output);
void prepare_peer_moe(uint32_t attention_tp);
void launch_peer_moe(void *stream, void *input, void *inverse, void *ids, void *weights,
                     void *shared, void *output, void *local, void *table, uint32_t rank,
                     uint32_t rows, uint32_t topk, bool finalize, uint32_t attention_tp,
                     bool routed_bf16 = false);
void prepare_peer_mhc(uint32_t world, uint32_t attention_tp);
void launch_peer_mhc(void *stream, void *input, void *residual, void *pre, void *post, void *comb,
                     void *weight, void *updated, void *output, void *local, void *table,
                     void *inverse, void *ids, void *weights, void *shared, uint32_t rank,
                     uint32_t rows, uint32_t topk, uint32_t world, bool normalize,
                     uint32_t attention_tp, bool routed_bf16 = false);
