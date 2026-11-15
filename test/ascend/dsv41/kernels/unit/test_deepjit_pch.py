"""Actual Bisheng PCH compilation, without a device context or NPU launch."""

import json
import os
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def pch_probe(tmp_path_factory):
    compiler = shutil.which("bisheng")
    if not compiler or not shutil.which("g++-13"):
        pytest.skip("Bisheng and GCC 13 are required for the CPU compiler probe")
    directory = tmp_path_factory.mktemp("deepjit_sdk_pch")
    source = directory / "probe.cpp"
    source.write_text(r"""
#include <deep_jit/backend/ascend/pch.hpp>
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    try {
        std::ifstream input(argv[1]);
        std::vector<std::string> base, defines;
        std::string token;
        while (input >> std::quoted(token)) base.push_back(token);
        for (int i = 4; i < argc; ++i) defines.emplace_back(argv[i]);
        const auto linker = std::filesystem::path(base[0]).parent_path() / "ld.lld";
        deep_jit::ascend::SdkPch::compile(base, defines, "unit-sdk-pch", deep_jit::Env("DSV41"),
            argv[2], argv[3], linker, false);
    } catch (const std::exception& error) {
        std::cerr << error.what(); return 1;
    }
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
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable, compiler


def test_pch_preserves_binary_and_reuses_sdk_across_aggregate_macros(
    pch_probe, tmp_path
):
    executable, compiler = pch_probe
    source = tmp_path / "kernel.asc"
    source.write_text("""#include <kernel_operator.h>
#include <c_api/asc_simd.h>
constexpr int kValues[] = PROJECT_VALUES;
extern "C" __global__ __aicore__ void pch_macro_probe(GM_ADDR out) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    reinterpret_cast<__gm__ int32_t*>(out)[0] = kValues[0] + kValues[1];
}
""")
    base = [
        compiler,
        "-O2",
        "--npu-arch=dav-2201",
        "--cce-aicore-only",
        "--cce-auto-sync=off",
    ]
    flags = tmp_path / "base.flags"
    flags.write_text("\n".join(json.dumps(value) for value in base))
    env = {
        **os.environ,
        "DSV41_JIT_CACHE_DIR": str(tmp_path / "jit"),
        "DJ_DISABLE_CACHE_FALLBACK": "1",
    }
    first, second = "-DPROJECT_VALUES={11,17}", "-DPROJECT_VALUES={31,53}"
    baseline, actual, changed = (
        tmp_path / name for name in ("baseline.o", "actual.o", "changed.o")
    )
    subprocess.run(
        [*base, first, str(source), "-o", str(baseline)],
        env=env,
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        [str(executable), str(flags), str(source), str(actual), first],
        env=env,
        check=True,
        capture_output=True,
        timeout=60,
    )
    assert actual.read_bytes() == baseline.read_bytes()
    files = sorted((tmp_path / "jit/cache").glob("ascend_sdk_pch.*/*.pch"))
    assert len(files) == 2
    timestamps = {p: p.stat().st_mtime_ns for p in files}
    subprocess.run(
        [str(executable), str(flags), str(source), str(changed), second],
        env=env,
        check=True,
        capture_output=True,
        timeout=60,
    )
    assert changed.read_bytes() != actual.read_bytes(), (
        "project defines were frozen in the PCH"
    )
    assert {p: p.stat().st_mtime_ns for p in files} == timestamps
