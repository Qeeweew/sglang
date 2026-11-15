#include "launch.h"
#include "actual_entry.h"
#include "torch.h"
namespace {
enum class Stage : uint64_t { Prepare, Rank };
using Entries = native_jit::ActualEntries<4>;
native_jit::Kernel &entry(Stage stage, int64_t gamma, int64_t columns,
                          uint32_t workers) {
    static thread_local Entries entries;
    const Entries::Key key{uint64_t(stage), uint64_t(gamma), uint64_t(columns),
                           workers};
    return entries.get(key, [&] {
        std::string tag = stage == Stage::Prepare ? "dspark_prepare" : "dspark_rank";
        for (auto value : key)
            tag += "_" + std::to_string(value);
        std::vector<std::string> flags{"--cce-auto-sync=off",
                                       "-DDSPARK_SCHEDULE_GAMMA=" + std::to_string(gamma),
                                       "-DDSPARK_SCHEDULE_COLUMNS=" + std::to_string(columns),
                                       "-DDSPARK_SCHEDULE_WORKERS=" + std::to_string(workers)};
        if (stage == Stage::Prepare)
            flags.emplace_back("-DDSPARK_SCHEDULE_PREPARE=1");
        return std::make_unique<native_jit::Kernel>(
            tag.c_str(), "dspark_schedule/dspark_schedule.asc", std::move(flags), workers);
    });
}
} // namespace
extern "C" void dsv41_dspark_schedule_prepare_launch(void *stream, uint32_t blocks,
                                                     uint8_t *confidence, uint8_t *survival,
                                                     int64_t rows, int64_t gamma, int64_t columns,
                                                     int64_t /* stride */, float eps) {
    entry(Stage::Prepare, gamma, columns, blocks)
        .launch_persistent(stream, (rows + dspark_schedule::kRequestGroup - 1) /
                                       dspark_schedule::kRequestGroup, confidence, survival, eps, rows);
}
extern "C" void dsv41_dspark_schedule_finish_launch(void *stream, uint32_t blocks,
                                                    uint8_t *survival, uint8_t *output,
                                                    int64_t rows, int64_t columns,
                                                    int64_t /* stride */, int64_t budget,
                                                    int64_t minimum, int64_t maximum) {
    entry(Stage::Rank, 0, columns, blocks)
        .launch_persistent(stream, (rows + dspark_schedule::kRequestGroup - 1) /
                                       dspark_schedule::kRequestGroup, survival, output,
                budget, minimum, maximum, rows);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_dspark_schedule(int rows, int gamma, int columns, int workers, bool "
          "prepare=True) -> ()",
          [](int64_t rows, int64_t gamma, int64_t columns, int64_t workers, bool prepare) {
              TORCH_CHECK_VALUE(rows > 0 && gamma > 0 && columns > 0 && columns <= gamma &&
                                    workers > 0 &&
                                    workers <= native_host::core_count(c10_npu::current_device()),
                                "actual scheduler geometry and worker plan required");
              entry(prepare ? Stage::Prepare : Stage::Rank, prepare ? gamma : 0, columns,
                    uint32_t(workers))
                  .prepare();
          });
}
