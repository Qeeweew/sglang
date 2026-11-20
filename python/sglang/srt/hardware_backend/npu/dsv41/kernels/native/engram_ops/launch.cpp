#include "launch.h"
#include "layout.h"
#include "actual_entry.h"
#include "torch.h"
#include <array>
#include <sstream>
namespace {
enum class Stage : int64_t { Hash, Commit, Gate };
// Each stage uses only its fields; unneeded fields are zero, never specialization axes.
struct Geometry {
    Stage stage;
    int64_t history_rows, vocab, pad, image, mm_shift;
    int64_t layers, history_tokens, heads, width, branches, workers;
    bool streaming = false;
    auto operator<=>(const Geometry &) const = default;
};
native_jit::Kernel &entry(const Geometry &g) {
    static thread_local native_jit::KernelCache<Geometry> entries;
    return entries.get(g, [&] {
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        auto field = [&](const char *name, int64_t value) {
            flags.emplace_back(std::string("-DENGRAM_OPS_") + name + "=" + std::to_string(value));
        };
        field("WORKERS", g.workers);
        const char *tag;
        if (g.stage == Stage::Hash) {
            tag = "engram_hash";
            field("HASH", 1);
            field("HISTORY_ROWS", g.history_rows);
            field("VOCAB", g.vocab);
            field("PAD", g.pad);
            field("IMAGE", g.image);
            field("MM_SHIFT", g.mm_shift);
            field("LAYERS", g.layers);
            field("HISTORY_TOKENS", g.history_tokens);
            field("HEADS", g.heads);
        } else if (g.stage == Stage::Commit) {
            tag = "engram_commit";
            field("COMMIT", 1);
            field("COMMIT_STREAMING", g.streaming);
            field("HISTORY_ROWS", g.history_rows);
            field("HISTORY_TOKENS", g.history_tokens);
        } else {
            tag = "engram_gate";
            field("WIDTH", g.width);
            field("BRANCHES", g.branches);
            // Preserve the former host double sqrt -> FP32 rounding exactly.
            std::ostringstream inverse;
            inverse << std::hexfloat << float(1.0 / std::sqrt(double(g.width))) << 'f';
            flags.emplace_back("-DENGRAM_OPS_INV_SQRT_WIDTH=" + inverse.str());
        }
        return std::make_unique<native_jit::Kernel>(tag, "engram_ops/engram_ops.asc", flags,
                                                    uint32_t(g.workers));
    });
}
} // namespace
extern "C" void dsv41_engram_hash_launch(
    void *stream, uint32_t blocks, uint8_t *ids, uint8_t *positions, uint8_t *cu, uint8_t *slots,
    uint8_t *history, uint8_t *active, uint8_t *map, uint8_t *multipliers, uint8_t *primes,
    uint8_t *offsets, uint8_t *output, int64_t rows, int64_t batch, int64_t history_rows,
    int64_t vocab, int64_t pad, int64_t image, int64_t mm_shift, uint32_t layers,
    uint32_t history_tokens, uint32_t heads, uint32_t /* hash_columns */) {
    entry({Stage::Hash, history_rows, vocab, pad, image, mm_shift, layers,
           history_tokens, heads, 0, 0, blocks})
        .launch_persistent(stream, uint64_t(rows) * layers, ids, positions, cu, slots,
            history, active, map, multipliers,
                           primes, offsets, output, rows, batch);
}
extern "C" void dsv41_engram_commit_launch(void *stream, uint32_t blocks, uint8_t *ids, uint8_t *cu,
                                           uint8_t *slots, uint8_t *accepted, uint8_t *history,
                                           int64_t rows, int64_t batch, int64_t history_rows,
                                           uint32_t history_tokens) {
    const auto dimensions =
        engram_ops::commit_dimensions(rows, batch, history_rows, history_tokens);
    entry({Stage::Commit, dimensions.history_rows, 0, 0, 0, 0, 0,
           history_tokens, 0, 0, 0, blocks, batch > blocks})
        .launch_persistent(stream, batch, ids, cu, slots, accepted, history,
            dimensions.rows, batch);
}
extern "C" void dsv41_engram_gate_launch(void *stream, uint32_t blocks, uint8_t *hidden,
                                         uint8_t *kv, uint8_t *query_weight, uint8_t *key_weight,
                                         uint8_t *active, uint8_t *output, int64_t tokens,
                                         int64_t width, int64_t branches,
                                         float /* inv_sqrt_width */) {
    entry({Stage::Gate, 0, 0, 0, 0, 0, 0, 0, 0, width, branches, blocks})
        .launch_persistent(stream, uint64_t(tokens) * branches, hidden, kv, query_weight,
            key_weight, active, output, tokens);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def(
        "precompile_engram_ops(str stage, int rows, int workers, int batch=0, int history_rows=0, "
        "int vocab=0, int pad=0, int image=-1, int mm_shift=1000000, int layers=0, "
        "int history_tokens=0, int heads=0, int width=0, int branches=0) -> ()",
        [](const std::string &stage, int64_t rows, int64_t workers, int64_t batch,
           int64_t history_rows, int64_t vocab, int64_t pad, int64_t image, int64_t mm_shift,
           int64_t layers, int64_t history_tokens, int64_t heads, int64_t width, int64_t branches) {
            TORCH_CHECK(rows >= 0 && workers > 0 &&
                            workers <= native_host::core_count(c10_npu::current_device()),
                        "invalid actual rows/workers");
            if (stage == "hash") {
                TORCH_CHECK(batch >= 0 && history_rows >= 0 && vocab > 0 && layers > 0 &&
                                layers <= UINT32_MAX && history_tokens > 0 && heads > 0 &&
                                uint64_t(history_tokens) * heads <=
                                    arch2201::kUbBytes / sizeof(int64_t),
                            "invalid hash geometry");
                entry({Stage::Hash, history_rows, vocab, pad, image, mm_shift, layers,
                       history_tokens, heads, 0, 0, workers})
                    .prepare();
            } else if (stage == "commit") {
                TORCH_CHECK(batch > 0 && history_rows >= 0 && history_tokens > 0 &&
                                history_tokens <= arch2201::kUbBytes / sizeof(int32_t),
                            "invalid commit geometry");
                const auto d =
                    engram_ops::commit_dimensions(rows, batch, history_rows, history_tokens);
                entry({Stage::Commit, d.history_rows, 0, 0, 0, 0, 0, history_tokens,
                       0, 0, 0, workers, batch > workers})
                    .prepare();
            } else {
                TORCH_CHECK(stage == "gate" && width >= 64 && width % 64 == 0 && branches > 0 &&
                                width <= (arch2201::kUbBytes -
                                          engram_ops::GateLayout::kStatBytes) /
                                             engram_ops::GateLayout::kBytesPerColumn,
                            "invalid gate geometry");
                entry({Stage::Gate, 0, 0, 0, 0, 0, 0, 0, 0, width, branches, workers})
                    .prepare();
            }
        });
}
