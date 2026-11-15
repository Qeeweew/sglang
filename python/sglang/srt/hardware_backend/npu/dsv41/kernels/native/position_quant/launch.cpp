#include "launch.h"
#include "params.h"
#include "jit.h"
#include <algorithm>
#include <map>
#include <memory>
#include <tuple>
#include <torch/library.h>
namespace {
struct PositionKey {
    uint32_t operation = 0, dtype = 0, heads = 0, dim = 0, rotary_dim = 0, mode = 0;
    bool inverse = false, inplace = false, has_trig = false, with_scales = false,
         with_slots = false;
    int64_t bound = 0;
    uint32_t tile_heads = 0, tile_tokens = 0;
    uint32_t workers = 0;
    auto fields() const {
        return std::tie(operation, dtype, heads, dim, rotary_dim, mode, inverse, inplace, has_trig,
                        with_scales, with_slots, bound, tile_heads, tile_tokens, workers);
    }
    bool operator<(const PositionKey &other) const {
        return fields() < other.fields();
    }
    bool operator==(const PositionKey &other) const {
        return fields() == other.fields();
    }
};
native_jit::Kernel &entry(const PositionKey &k) {
    static thread_local std::map<PositionKey, std::unique_ptr<native_jit::Kernel>> cache;
    static thread_local PositionKey last_key{};
    static thread_local native_jit::Kernel *last = nullptr;
    if (last && last_key == k)
        return *last;
    auto &result = cache[k];
    if (!result) {
        const char *name = k.operation == 0   ? "position_rope"
                           : k.operation == 1 ? "position_quant"
                           : k.operation == 3 ? "rope_cache_gather"
                           : k.dtype == 0     ? "cache_scatter_int8"
                                              : "cache_scatter_bf16";
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        if (k.operation == 0)
            flags.emplace_back("-DPOSITION_QUANT_ROPE=1");
        if (k.operation == 2)
            flags.emplace_back(k.dtype == 0 ? "-DPOSITION_QUANT_SCATTER_INT8=1"
                                            : "-DPOSITION_QUANT_SCATTER_BF16=1");
        if (k.operation == 3)
            flags.emplace_back("-DPOSITION_QUANT_GATHER=1");
        const auto add = [&](const char *label, auto value) {
            flags.emplace_back(std::string("-DPOSITION_QUANT_") + label + "=" +
                               std::to_string(value));
        };
        add("HEADS", k.heads);
        add("DIM", k.dim);
        add("ROTARY_DIM", k.rotary_dim);
        add("MODE", k.mode);
        add("INVERSE", k.inverse);
        add("INPLACE", k.inplace);
        add("HAS_TRIG", k.has_trig);
        add("WITH_SCALES", k.with_scales);
        add("WITH_SLOTS", k.with_slots);
        add("BOUND", k.bound);
        add("WORKERS", k.workers);
        add("TILE_HEADS", k.tile_heads);
        add("TILE_TOKENS", k.tile_tokens);
        const auto workers = k.workers;
        result = std::make_unique<native_jit::Kernel>(name, "position_quant/position_quant.asc",
                                                      flags, workers);
    }
    last_key = k;
    last = result.get();
    return *result;
}
PositionKey position_key(uint32_t heads, uint32_t dim, uint32_t rotary_dim, uint32_t mode,
                         bool inverse, bool inplace, bool has_trig, uint32_t workers) {
    PositionKey key;
    key.operation = mode == 0 ? 0 : 1;
    key.heads = heads;
    key.dim = dim;
    key.rotary_dim = rotary_dim;
    key.mode = mode;
    key.inverse = inverse;
    key.inplace = mode == 0 && inplace;
    key.has_trig = has_trig;
    key.workers = workers;
    return key;
}
PositionKey plan_rows(PositionKey key, int64_t rows) {
    const auto budget = key.workers;
    if (key.operation == 0) {
        const auto plan = position_quant::rope_plan(rows, key.heads, budget);
        key.tile_heads = plan.heads;
        key.tile_tokens = plan.tokens;
    }
    return key;
}
} // namespace
extern "C" void dsv41_position_quant_launch(void *stream, uint32_t blocks, uint8_t *x, uint8_t *cos,
                                            uint8_t *sin, uint8_t *y, uint8_t *scales,
                                            uint8_t *tables, int64_t rows, uint32_t heads,
                                            uint32_t dim, uint32_t mode, bool inverse,
                                            uint32_t rotary_dim) {
    auto key = position_key(heads, dim, rotary_dim, mode, inverse, x == y, cos != nullptr, blocks);
    key = plan_rows(key, rows);
    const uint64_t jobs = mode == 0 ? position_quant::rope_plan(rows, heads, blocks).jobs
        : (rows + position_quant::kScaleGroupRows - 1) / position_quant::kScaleGroupRows;
    if (mode == 0)
        entry(key).launch_persistent(stream, jobs, x, cos, sin, y, tables, rows);
    else
        entry(key).launch_persistent(stream, jobs, x, cos, sin, y, scales, tables, rows);
}
extern "C" void dsv41_cache_scatter_launch(void *stream, uint32_t blocks, uint8_t *values,
                                           uint8_t *scales, uint8_t *slots, uint8_t *cache,
                                           uint8_t *cache_scales, int64_t rows, int64_t capacity,
                                           uint32_t dim, uint32_t dtype) {
    PositionKey key;
    key.operation = 2;
    key.dtype = dtype;
    key.dim = dim;
    key.bound = capacity;
    key.with_scales = cache_scales != nullptr;
    key.workers = blocks;
    key = plan_rows(key, rows);
    // Ownership follows destination cache lines, not source row ordinals.
    const uint64_t owners = key.with_scales
        ? (capacity + position_quant::kScaleGroupRows - 1) / position_quant::kScaleGroupRows
        : capacity;
    entry(key).launch_persistent(stream, owners, values, scales, slots, cache, cache_scales, rows);
}
extern "C" void dsv41_rope_cache_gather_launch(void *stream, uint32_t blocks, uint8_t *cos,
                                               uint8_t *sin, uint8_t *positions,
                                               uint8_t *input_slots, uint8_t *cos_out,
                                               uint8_t *sin_out, uint8_t *valid_out,
                                               uint8_t *slots_out, int64_t rows, int64_t cache_rows,
                                               uint32_t trig_columns) {
    PositionKey key;
    key.operation = 3;
    key.dim = trig_columns;
    key.bound = cache_rows;
    key.with_slots = input_slots != nullptr;
    key.workers = blocks;
    key = plan_rows(key, rows);
    entry(key).launch_persistent(stream,
        (rows + position_quant::kGatherGroupRows - 1) / position_quant::kGatherGroupRows, cos, sin, positions, input_slots, cos_out, sin_out,
                                 valid_out, slots_out, rows);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_position_quant(int operation, int rows, int worker_budget, int heads=0, int "
          "dim=0, "
          "int rotary_dim=0, int mode=0, int dtype=0, bool inverse=False, bool inplace=False, "
          "bool has_trig=False, bool with_scales=False, bool with_slots=False, int bound=-1) -> ()",
          [](int64_t operation, int64_t rows, int64_t workers, int64_t heads, int64_t dim,
             int64_t rotary_dim, int64_t mode, int64_t dtype, bool inverse, bool inplace,
             bool has_trig, bool with_scales, bool with_slots, int64_t bound) {
              TORCH_CHECK(operation >= 0 && operation <= 3 && rows > 0 && workers > 0 &&
                              workers <= UINT32_MAX && dim > 0,
                          "actual position layout and worker plan required");
              if (operation <= 1)
                  TORCH_CHECK(heads > 0 && heads <= UINT32_MAX && rotary_dim > 0 &&
                                  rotary_dim <= 64 && rotary_dim % 16 == 0 && dim <= 2048 &&
                                  dim % 16 == 0 && (!has_trig || dim >= rotary_dim) &&
                                  (operation == 0 ? mode == 0 && has_trig : mode == 1 || mode == 2),
                              "actual heads, rotary layout and operation mode required");
              if (operation >= 2)
                  TORCH_CHECK(bound >= 0, "explicit physical cache bound required (zero is valid)");
              if (operation == 2)
                  TORCH_CHECK((dtype == 0 || dtype == 1) && dim <= 2048 &&
                                  dim % (dtype == 0 ? 32 : 16) == 0,
                              "invalid cache payload layout");
              if (operation == 3)
                  TORCH_CHECK(dim >= 8 && dim <= 32 && dim % 8 == 0, "invalid trig columns");
              PositionKey key;
              if (operation <= 1) {
                  key = position_key(uint32_t(heads), uint32_t(dim), uint32_t(rotary_dim),
                                     uint32_t(mode), inverse, inplace, has_trig, uint32_t(workers));
              } else {
                  key.operation = uint32_t(operation);
                  key.workers = uint32_t(workers);
                  key.dim = uint32_t(dim);
                  key.bound = bound;
                  if (operation == 2) {
                      key.dtype = uint32_t(dtype);
                      key.with_scales = with_scales;
                  } else {
                      key.with_slots = with_slots;
                  }
              }
              key = plan_rows(key, rows);
              entry(key).prepare();
          });
}
