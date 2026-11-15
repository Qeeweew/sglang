"""Verify checkout identity and the published native extension before serving."""

import argparse
import json
from pathlib import Path

import sglang

from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.builders.native import _sources

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

repository = Path(__file__).resolve().parents[3]
source = Path(sglang.__file__).resolve()
if not source.is_relative_to(repository / "python"):
    raise RuntimeError(
        f"SGLang was imported from {source}; prepend this checkout's python directory to PYTHONPATH"
    )
from sglang.srt.parser.deepseek_recipe_renderer import load_deepseek_recipe

load_deepseek_recipe()
publication = json.loads((native_library_path().parent / "manifest.json").read_text())
if _sources(publication["components"]) != publication["sources"]:
    raise RuntimeError(
        "Native sources or vendored headers changed; rebuild the publication before serving"
    )
operators = load_library()
if not args.quiet:
    print(
        f"Environment check passed (SGLang: {source}, native ABI: {operators.abi()}).",
        flush=True,
    )
