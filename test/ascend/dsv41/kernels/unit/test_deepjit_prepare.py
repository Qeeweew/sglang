"""Preparation drains parallel work and leaves reusable state after errors."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.precompile import (
    prepare_batch,
)


def test_nested_prepare_batch_has_one_native_lifetime():
    operators = SimpleNamespace(begin_prepare_batch=Mock(), end_prepare_batch=Mock())
    with prepare_batch(operators):
        with prepare_batch(operators):
            operators.end_prepare_batch.assert_not_called()
    operators.begin_prepare_batch.assert_called_once()
    operators.end_prepare_batch.assert_called_once()


def test_host_error_still_drains_native_batch():
    operators = SimpleNamespace(begin_prepare_batch=Mock(), end_prepare_batch=Mock())
    with pytest.raises(ValueError, match="host validation"):
        with prepare_batch(operators):
            raise ValueError("host validation")
    operators.end_prepare_batch.assert_called_once()
    with prepare_batch(operators):
        pass
    assert operators.begin_prepare_batch.call_count == 2


def test_compile_error_does_not_leave_python_scope_active():
    operators = SimpleNamespace(begin_prepare_batch=Mock(), end_prepare_batch=Mock())
    operators.end_prepare_batch.side_effect = RuntimeError("compiler failed")
    with pytest.raises(RuntimeError, match="compiler failed"):
        with prepare_batch(operators):
            pass
    operators.end_prepare_batch.side_effect = None
    with prepare_batch(operators):
        pass
    assert operators.begin_prepare_batch.call_count == 2


@pytest.fixture(scope="module")
def queue_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("deepjit_prepare")
    source = directory / "probe.cpp"
    source.write_text(r"""
#include <deep_jit/runtime/precompiler.hpp>
#include <atomic>
#include <iostream>
#include <latch>

int main(int argc, char** argv) {
    const bool fail = argc > 1 && std::string(argv[1]) == "fail";
    const bool debug = argc > 2 && std::string(argv[2]) == "debug";
    std::atomic<int> completed = 0;
    std::latch first_wave(4);
    deep_jit::Precompiler pool(4, debug);
    for (int i = 0; i < 8; ++i) {
        for (int duplicate = 0; duplicate < 2; ++duplicate) {
            pool.enqueue(std::to_string(i), [&, i] {
                if (i < 4) {
                    first_wave.count_down();
                    first_wave.wait(); // Sequential execution cannot pass.
                }
                ++completed;
                if (fail && i == 0) throw std::runtime_error("compile failure");
            });
        }
    }
    bool caught = false;
    try { pool.finish(); }
    catch (const std::runtime_error&) { caught = true; }
    if (completed != 8 || caught != fail) return 1;
    pool.finish(); // No repeated future.get or destructor termination.
    std::cout << "drained 8 unique jobs\n";
}
""")
    from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import deepjit_root

    include = deepjit_root() / "include"
    executable = directory / "probe"
    subprocess.run(
        [
            "g++-13",
            "-std=c++20",
            "-O2",
            "-pthread",
            "-I",
            str(include),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_parallel_queue_deduplicates_and_drains_before_error(queue_probe, fail, debug):
    result = subprocess.run(
        [str(queue_probe), "fail" if fail else "run", *(["debug"] if debug else [])],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert result.stdout == "drained 8 unique jobs\n"
    if debug:
        assert "[DeepJIT]" in result.stderr
    else:
        assert result.stderr == ""
