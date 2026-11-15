"""Exercise native preparation locks with the unmodified DeepJIT disk cache."""

import os
import subprocess
import time

import pytest


@pytest.fixture(scope="module")
def compiler_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("deepjit_process_cache")
    source = directory / "probe.cpp"
    source.write_text(r"""
#include <deep_jit/cache/compile_lock.hpp>
#include <deep_jit/cache/disk.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

int main(int argc, char** argv) {
    const std::filesystem::path root = argv[1];
    const std::string key = argv[2], mode = argv[3];
    std::cin.get(); // Parent releases all competing processes together.
    deep_jit::DiskCache cache({root});
    bool hit;
    size_t root_index;
    {
        auto initial = cache.entry("probe", key);
        hit = initial.hit;
        root_index = initial.root_index;
    }
    std::unique_ptr<deep_jit::CompileLock> lock;
    if (!hit)
        lock = std::make_unique<deep_jit::CompileLock>(cache.paths[root_index], key);
    // This is DeepJIT's lookup after the integration lock, before compilation.
    auto entry = cache.entry("probe", key);
    if (!entry.hit) {
        // A compiler invocation, including one that subsequently fails.
        const int fd = ::open((root / "compiles").c_str(), O_CREAT | O_WRONLY | O_APPEND, 0600);
        const auto line = key + "\n";
        if (fd < 0 || ::write(fd, line.data(), line.size()) != long(line.size())) return 18;
        ::close(fd);
        deep_jit::write_file_sync(root / ("active." + key), "");
        if (mode == "hold") {
            for (;;) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (mode == "paired") {
            const auto other = root / (key == "a" ? "active.b" : "active.a");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!std::filesystem::exists(other)) {
                if (std::chrono::steady_clock::now() > deadline) return 19;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (mode == "fail") return 17;
        deep_jit::write_file_sync(entry.path / "artifact", key);
        entry.commit();
    }
    return deep_jit::read(entry.path / "artifact") == key ? 0 : 20;
}
""")
    executable = directory / "probe"
    from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT, deepjit_root

    include = deepjit_root() / "include"
    subprocess.run(
        [
            "g++-13",
            "-std=c++20",
            "-O2",
            "-pthread",
            "-I",
            str(include),
            "-I",
            str(ROOT / "native"),
            str(source),
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def start(probe, root, key="same", mode="run"):
    # Override inherited production options for deterministic CPU-only tests.
    env = {**os.environ, "DJ_DISABLE_CACHE_FALLBACK": "1"}
    process = subprocess.Popen(
        [str(probe), str(root), key, mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return process


def release(process):
    process.stdin.write(b"\n")
    process.stdin.flush()


def finish(process, expected=0):
    try:
        process.wait(timeout=15)
        assert process.returncode == expected, process.stderr.read().decode()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_eight_processes_compile_once_and_reuse_warm_cache(compiler_probe, tmp_path):
    processes = [start(compiler_probe, tmp_path) for _ in range(8)]
    for process in processes:
        release(process)
    for process in processes:
        finish(process)
    assert (tmp_path / "compiles").read_text().splitlines() == ["same"]
    warm = start(compiler_probe, tmp_path)
    release(warm)
    finish(warm)
    assert (tmp_path / "compiles").read_text().splitlines() == ["same"]
    assert not list((tmp_path / "tmp").iterdir())


def test_distinct_keys_compile_concurrently(compiler_probe, tmp_path):
    processes = [start(compiler_probe, tmp_path, key, "paired") for key in ("a", "b")]
    for process in processes:
        release(process)
    for process in processes:
        finish(process)
    assert sorted((tmp_path / "compiles").read_text().splitlines()) == ["a", "b"]


def test_failed_compile_releases_lock_and_can_retry(compiler_probe, tmp_path):
    failed = start(compiler_probe, tmp_path, mode="fail")
    release(failed)
    finish(failed, expected=17)
    assert not list((tmp_path / "cache").glob("*/.committed"))
    retry = start(compiler_probe, tmp_path)
    release(retry)
    finish(retry)
    assert (tmp_path / "compiles").read_text().splitlines() == ["same", "same"]


def test_killed_compiler_does_not_leave_stale_lock(compiler_probe, tmp_path):
    killed = start(compiler_probe, tmp_path, mode="hold")
    release(killed)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "active.same").exists():
            assert killed.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        killed.kill()
        killed.wait()
    retry = start(compiler_probe, tmp_path)
    release(retry)
    finish(retry)
    assert (tmp_path / "compiles").read_text().splitlines() == ["same", "same"]
