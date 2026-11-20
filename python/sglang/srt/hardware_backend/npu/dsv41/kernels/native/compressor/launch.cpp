#include "launch.h"
#include "actual_entry.h"
#include <compressor/params.h>
#include <array>
#include <algorithm>
#include <memory>
#include <torch/library.h>
namespace {
// ratio, payload width, persistent state capacity, selected launch workers.
using CompressorKey = std::array<int64_t, 4>;
native_jit::Kernel &entry(const CompressorKey &key, bool commit) {
    static thread_local native_jit::KernelCache<std::pair<CompressorKey, bool>> entries;
    return entries.get(std::make_pair(key, commit), [&] {
        const std::string name = commit ? "compressor_commit" : "compressor_prepare";
        std::vector<std::string> flags{"--cce-auto-sync=off",
                                       "-DCOMPRESSOR_RATIO=" + std::to_string(key[0]),
                                       "-DCOMPRESSOR_WIDTH=" + std::to_string(key[1]),
                                       "-DCOMPRESSOR_SLOTS=" + std::to_string(key[2]),
                                       "-DCOMPRESSOR_WORKERS=" + std::to_string(key[3])};
        if (commit)
            flags.emplace_back("-DCOMPRESSOR_COMMIT=1");
        return std::make_unique<native_jit::Kernel>(name.c_str(), "compressor/compressor.asc",
                                                    flags, uint32_t(key[3]));
    });
}
} // namespace
extern "C" void dsv41_compressor_prepare_launch(void *stream, uint32_t blocks, uint8_t *kv,
                                                uint8_t *scores, uint8_t *gamma, uint8_t *cu,
                                                uint8_t *requests, uint8_t *pending,
                                                uint8_t *lengths, uint8_t *pooled, uint8_t *latent,
                                                uint8_t *positions, uint8_t *rope_positions,
                                                uint8_t *starts, int64_t tokens, int64_t batches,
                                                int64_t slots, int64_t ratio, int64_t width) {
    entry({ratio, width, slots, blocks}, false)
        .launch_persistent(stream, std::max(tokens, batches), kv, scores, gamma,
            cu, requests, pending, lengths, pooled,
                           latent, positions, rope_positions, starts, tokens, batches);
}
extern "C" void dsv41_compressor_commit_launch(void *stream, uint32_t blocks, uint8_t *kv,
                                               uint8_t *scores, uint8_t *cu, uint8_t *requests,
                                               uint8_t *starts, uint8_t *accepted, uint8_t *pending,
                                               uint8_t *lengths, uint8_t *positions,
                                               uint8_t *rope_positions, uint8_t *status,
                                               int64_t batches, int64_t slots, int64_t ratio,
                                               int64_t width) {
    entry({ratio, width, slots, blocks}, true)
        .launch_persistent(stream, batches, kv, scores, cu, requests, starts,
            accepted, pending, lengths,
                           positions, rope_positions, status, batches);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_compressor(int ratio, int width, int slots, int workers, int batches, int "
          "tokens=0, bool commit=False) "
          "-> ()",
          [](int64_t ratio, int64_t width, int64_t slots, int64_t workers, int64_t batches,
             int64_t tokens, bool commit) {
              TORCH_CHECK((ratio == 1 || ratio == 2) && width > 0 &&
                              width <= compressor::kWidthCapacityElements &&
                              width % compressor::kWidthAlignmentElements == 0 &&
                              slots > 0 && batches >= 0 && tokens >= 0 && workers > 0 &&
                              workers <= UINT32_MAX,
                          "actual aligned compressor geometry and worker plan required");
              entry({ratio, width, slots, workers}, commit).prepare();
          });
}
