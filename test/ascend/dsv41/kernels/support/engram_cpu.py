"""CPU Engram decoder shared by independent numerical checks."""

import ctypes
import subprocess

from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT


def cpu_function(library_directory):
    library_directory.mkdir(parents=True, exist_ok=True)
    output = library_directory / "libdsv41_engram_cpu.so"
    source = ROOT / "tests/support/engram_cpu.cpp"
    if not output.exists() or source.stat().st_mtime > output.stat().st_mtime:
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-shared",
                str(source),
                "-o",
                str(output),
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(output.resolve()))
    function = lib.dsv41_engram_cpu
    function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2
    function.restype = None
    return function
