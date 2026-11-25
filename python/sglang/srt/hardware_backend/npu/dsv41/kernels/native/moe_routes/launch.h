#pragma once
#include <cstdint>
extern "C" void dsv41_route_gather(void *, uint32_t, void *, void *, void *, void *, void *,
                                   int64_t, int64_t, int64_t, int64_t);
extern "C" void dsv41_route_combine(void *, uint32_t, void *, void *, void *, void *, void *,
                                    int64_t, int64_t, int64_t, bool);
void dsv41_route_gather_a8(void *, uint32_t, void *, void *, void *, void *, void *, void *, void *,
                           int64_t, int64_t, int64_t);
