#pragma once
#include <cstdint>
#include "params.h"
extern "C" void dsv41_qli_launch(void *stream, uint32_t blocks, const qli::Config *config,
                                 uint8_t *q, uint8_t *k, uint8_t *w, uint8_t *qs, uint8_t *ks,
                                 uint8_t *cu, uint8_t *lengths, uint8_t *residual, uint8_t *pages,
                                 uint8_t *candidates, uint8_t *indices, uint8_t *candidate_out,
                                 uint8_t *workspace, uint8_t *positions);
