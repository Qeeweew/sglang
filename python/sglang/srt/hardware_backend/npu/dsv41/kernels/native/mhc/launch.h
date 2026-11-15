#pragma once
#include <mhc/params.h>
extern "C" void dsv41_mhc_project_launch(void *, uint32_t, uint8_t *, uint8_t *, uint8_t *,
                                         uint8_t *, uint8_t *, uint8_t *, mhc::ProjectParams,
                                         bool);
extern "C" void dsv41_mhc_stats_launch(void *, uint32_t, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                       uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                       uint8_t *, mhc::StatsParams, bool);
extern "C" void dsv41_mhc_combine_launch(void *, uint32_t, uint8_t *, uint8_t *, uint8_t *,
                                         uint8_t *, mhc::StateParams, bool, bool);
extern "C" void dsv41_mhc_post_launch(void *, uint32_t, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                      uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                      mhc::StateParams, bool);
extern "C" void dsv41_mhc_mean_launch(void *, uint32_t, uint8_t *, uint8_t *,
                                      mhc::StateParams);
