#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr unsigned signs = 0x9e3779b9;
constexpr int levels[] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
float nearest(float value) { return std::clamp(std::nearbyint(value), -8.f, 7.f); }
}

extern "C" int export_g32(const uint8_t *raw, const uint8_t *exps, uint8_t *packed,
                          uint8_t *metadata, int rows, int depth, int threads, float fold,
                          int projection) {
    int invalid = 0;
    const int groups = depth / 32;
#pragma omp parallel num_threads(threads) reduction(| : invalid)
    {
        std::vector<float> rotated(depth), step(groups), importance(groups);
        std::vector<int> factors(groups);
#pragma omp for
        for (int row = 0; row < rows; ++row) {
            const auto offset = int64_t(row) * depth;
            const int dest = projection == 2 ? row : row / 256 * 256 + row % 32 * 8 + row % 256 / 32;
            float maximum = 0;
            for (int block = 0; block < groups; ++block) {
                int16_t transformed[32];
                auto begin = offset + block * 32;
                for (int lane = 0; lane < 32; ++lane) {
                    int code = (raw[(begin + lane) / 2] >> ((lane & 1) * 4)) & 15;
                    transformed[lane] = (signs >> lane & 1) ? levels[code] : -levels[code];
                }
                for (int width = 1; width < 32; width *= 2)
                    for (int start = 0; start < 32; start += width * 2)
                        for (int lane = start; lane < start + width; ++lane) {
                            auto left = transformed[lane], right = transformed[lane + width];
                            transformed[lane] = left + right;
                            transformed[lane + width] = left - right;
                        }
                int exponent = exps[begin / 32];
                invalid |= exponent == 255;
                float scale = std::ldexp(0.5f / std::sqrt(32.f), exponent - 127);
                if (projection == 3 && !(signs >> (row % 32) & 1)) scale = -scale;
                float *values = rotated.data() + block * 32;
                for (int lane = 0; lane < 32; ++lane) values[lane] = scale * transformed[lane];
                auto bounds = std::minmax_element(values, values + 32);
                float delta = std::max(-*bounds.first / 8.f, *bounds.second / 7.f);
                float norm = 0;
                for (int iter = 0; iter < 2 && delta > 0; ++iter) {
                    float cross = 0; norm = 0;
                    float inv = 1.f / delta;
#pragma omp simd reduction(+ : cross, norm)
                    for (int lane = 0; lane < 32; ++lane) {
                        float code = nearest(values[lane] * inv);
                        cross += code * values[lane]; norm += code * code;
                    }
                    if (norm) delta = cross / norm;
                }
                step[block] = delta; importance[block] = norm;
                maximum = std::max(maximum, delta);
            }
            float best = std::numeric_limits<float>::infinity(), unit = 1.f;
            for (float seed : {0.5f, 0.75f, 1.f, 1.25f}) {
                if (!maximum) break;
                float trial = maximum * seed / 16;
                for (int iter = 0; iter < 4; ++iter) {
                    float cross = 0, norm = 0, loss = 0, inv = 1.f / trial;
#pragma omp simd reduction(+ : cross, norm, loss)
                    for (int block = 0; block < groups; ++block) {
                        float factor = std::clamp(std::nearbyint(step[block] * inv), 1.f, 16.f);
                        float diff = step[block] - trial * factor;
                        loss += importance[block] * diff * diff;
                        cross += importance[block] * factor * step[block];
                        norm += importance[block] * factor * factor;
                    }
                    if (loss < best) { best = loss; unit = trial; }
                    if (norm == 0 || trial == cross / norm) break;
                    trial = cross / norm;
                }
            }
            float cross = 0, norm = 0;
            auto meta = metadata + int64_t(dest) * (groups / 2 + 4);
            for (int block = 0; block < groups; ++block) {
                float *values = rotated.data() + block * 32;
                float min_loss = std::numeric_limits<float>::infinity();
                int winner = 1;
                for (int factor = 1; factor <= 16; ++factor) {
                    float delta = unit * factor, inv = 1.f / delta, loss = 0;
#pragma omp simd reduction(+ : loss)
                    for (int lane = 0; lane < 32; ++lane) {
                        float diff = values[lane] - delta * nearest(values[lane] * inv);
                        loss += diff * diff;
                    }
                    if (loss < min_loss) { min_loss = loss; winner = factor; }
                }
                factors[block] = winner;
                float inv = 1.f / (unit * winner);
                uint8_t codes[32];
#pragma omp simd reduction(+ : cross, norm)
                for (int lane = 0; lane < 32; ++lane) {
                    float code = nearest(values[lane] * inv);
                    codes[lane] = int(code) & 15;
                    float product = code * winner;
                    norm += product * product; cross += product * values[lane];
                }
                for (int lane = 0; lane < 16; ++lane)
                    packed[int64_t(dest) * (depth / 2) + block * 16 + lane] = codes[2 * lane] | (codes[2 * lane + 1] << 4);
                if (block & 1) meta[block / 2] = ((factors[block-1]-9)&15) | (((winner-9)&15)<<4);
            }
            unit = (norm ? cross / norm : 1.f) * fold;
            invalid |= !std::isfinite(unit) || !(unit > 0);
            std::memcpy(meta + groups / 2, &unit, 4);
        }
    }
    return invalid;
}
