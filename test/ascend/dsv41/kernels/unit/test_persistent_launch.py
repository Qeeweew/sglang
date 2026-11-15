"""Launch occupancy must preserve constexpr scheduling and device argument ABI."""

import subprocess
from pathlib import Path


def test_persistent_grid_and_partition_coverage(tmp_path):
    # Exercise the real host template without a device runtime. Only the
    # external launch sink and cold handle preparation are replaced.
    (tmp_path / "c10/util").mkdir(parents=True)
    (tmp_path / "c10/util/Exception.h").write_text(
        "#include <stdexcept>\n"
        '#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("plan"); } while (0)\n'
    )
    (tmp_path / "deep_jit/backend/ascend").mkdir(parents=True)
    (tmp_path / "deep_jit/backend/ascend/kernel.hpp").write_text(r"""
#pragma once
#include <cstdint>
#include <cassert>
namespace deep_jit::ascend {
struct LaunchOptions {
    void *stream;
    int num_blocks, num_ubuf_bytes, num_launch_timeout_secs;
};
inline LaunchOptions observed;
inline int launches = 0;
class Kernel {
public:
    void launch(const LaunchOptions &options, int *output, uint64_t rows) {
        assert(output && *output == 17 && rows == 123);
        observed = options;
        ++launches;
    }
};
}
""")
    source = tmp_path / "probe.cpp"
    source.write_text(r"""
#include <common/jit.h>
#include <router/layout.h>
#include <cassert>
#include <limits>
namespace native_jit {
bool preparation = false;
bool preparing_only() { return preparation; }
Kernel::Kernel(const char *, const char *, std::vector<std::string>, uint32_t workers)
    : workers_(workers) {}
const std::shared_ptr<deep_jit::ascend::Kernel> &Kernel::prepare() {
    static const auto handle = std::make_shared<deep_jit::ascend::Kernel>();
    return handle;
}
}
int main() {
    int value = 17;
    void *stream = &value;
    using namespace deep_jit::ascend;
    for (uint32_t maximum : {1, 7, 40}) {
        native_jit::Kernel entry("probe", "probe.asc", {}, maximum);
        for (uint64_t jobs = 0; jobs <= 3 * maximum + 1; ++jobs) {
            const int before = launches;
            entry.launch_persistent(stream, jobs, &value, uint64_t(123));
            if (jobs == 0) { assert(launches == before); continue; }
            assert(launches == before + 1);
            assert(observed.stream == stream && observed.num_ubuf_bytes == 0 &&
                   observed.num_launch_timeout_secs == 0);
            assert(observed.num_blocks >= 1 && observed.num_blocks <= int(maximum));
            std::vector<int> visits(jobs);
            for (int core = 0; core < observed.num_blocks; ++core)
                for (uint64_t job = core; job < jobs; job += maximum)
                    ++visits[job];
            for (int visited : visits) assert(visited == 1);
            if (jobs && jobs < maximum) assert(observed.num_blocks == int(jobs));
        }
        entry.launch_persistent(stream, std::numeric_limits<uint64_t>::max(), &value, uint64_t(123));
        assert(observed.num_blocks == int(maximum));
        // Router's contiguous partitions are not a cyclic job grid. Every
        // partition in the launched prefix must be nonempty, even at 41/40.
        for (uint32_t experts : {128, 384}) {
            const auto tile = router_layout::tile_rows(experts, 6);
            for (int64_t rows = 0; rows <= int64_t(3 * maximum + 1) * tile; ++rows) {
                const auto blocks = router_layout::task_blocks(rows, experts, 6, maximum);
                const int64_t groups = (rows + tile - 1) / tile;
                const int64_t per_core = (groups + maximum - 1) / maximum;
                if (groups == 0) { assert(blocks == 0); continue; }
                assert(blocks >= 1 && blocks <= maximum);
                std::vector<int> visits(groups);
                for (uint32_t core = 0; core < blocks; ++core) {
                    assert(core * per_core < groups);
                    for (int64_t group = core * per_core;
                         group < (core + 1) * per_core && group < groups; ++group)
                        ++visits[group];
                }
                for (int visited : visits) assert(visited == 1);
                assert(int64_t(blocks) * per_core >= groups);
            }
        }
        native_jit::preparation = true;
        const int before = launches;
        entry.launch_persistent(stream, 6, &value, uint64_t(123));
        assert(launches == before);
        native_jit::preparation = false;
    }
    native_jit::Kernel invalid("probe", "probe.asc");
    try { invalid.launch_persistent(stream, 6, &value, uint64_t(123)); return 1; }
    catch (const std::runtime_error &) {}
}
""")
    from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT

    root = ROOT
    executable = tmp_path / "probe"
    subprocess.run(
        [
            "g++-13",
            "-std=c++20",
            "-O2",
            "-I",
            str(tmp_path),
            "-I",
            str(root / "native"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=15)
