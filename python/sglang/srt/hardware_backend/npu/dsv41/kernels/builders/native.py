"""CMake builds one Torch-NPU extension; publish immutable binaries atomically."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from sglang.srt.hardware_backend.npu.dsv41.kernels.builders.common import (
    MAX_BUILD_SECONDS,
    _hash,
    _write_json,
    run,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import CANN, ROOT, deepjit_root

LIBRARY = "libdsv41_kernels.so"
COMPONENTS = tuple(
    item["name"]
    for item in json.loads((ROOT / "native/components.json").read_text())["components"]
)


def _sources(components=None):
    paths = [ROOT / "CMakeLists.txt"]
    for folder in ("cmake", "builders"):
        paths.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in {".pyc", ".pyo"}
        )
    native = ROOT / "native"
    roots = (
        (native,)
        if components is None
        else tuple(native / name for name in ("common", "host_memory", *components))
    )
    paths.extend(p for root in roots for p in root.rglob("*") if p.is_file())
    # Host-only build fixtures may keep their translation units at this level.
    if components is not None:
        paths.extend(p for p in native.iterdir() if p.is_file())
    # Include closure also covers old headers while operators are migrated.
    tracked = set(paths)
    pending = list(paths)
    includes = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)
    while pending:
        source = pending.pop()
        if source.suffix not in {".asc", ".cpp", ".h", ".hpp", ".inc"}:
            continue
        for name in includes.findall(source.read_text()):
            for base in (source.parent, native / "common", native):
                dependency = (base / name).resolve()
                if dependency.is_relative_to(native) and dependency.is_file():
                    if dependency not in tracked:
                        tracked.add(dependency)
                        pending.append(dependency)
                    break
    sources = {str(p.relative_to(ROOT)): _hash(p) for p in sorted(tracked)}
    vendor = deepjit_root()
    sources.update(
        {
            "3rdparty/deepjit/" + str(p.relative_to(vendor)): _hash(p)
            for p in vendor.rglob("*")
            if p.is_file()
        }
    )
    return sources


def source_path(name):
    prefix = "3rdparty/deepjit/"
    return (
        deepjit_root() / name[len(prefix) :] if name.startswith(prefix) else ROOT / name
    )


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _needs_clean_host(workspace, sources):
    """Make's timestamps are insufficient after restoring older source bytes.

    The ledger describes the actual successful build, independently of the
    published manifest. Missing/failed/externally changed builds are rebuilt.
    Ordinary edits newer than the library retain CMake's per-file incrementality.
    """
    if not workspace.exists():
        return False
    target = workspace / LIBRARY
    try:
        state = json.loads((workspace / "verified-inputs.json").read_text())
    except (OSError, ValueError):
        return True
    if (
        not isinstance(state, dict)
        or state.get("schema") != 1
        or state.get("source_root") != str(ROOT.resolve())
        or not isinstance(state.get("sources"), dict)
        or not target.is_file()
        or state.get("library_sha256") != _hash(target)
    ):
        return True
    previous = state["sources"]
    target_time = target.stat().st_mtime_ns
    # Headers may be host dependencies; a conservative clean is needed for
    # timestamp-preserving changes. Device .asc/docs never enter host builds.
    host_suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".inc", ".cmake"}
    for name, digest in sources.items():
        if name not in previous or previous[name] == digest:
            continue
        path = source_path(name)
        if path.suffix in host_suffixes or path.name == "CMakeLists.txt":
            if path.stat().st_mtime_ns <= target_time:
                return True
    return False


def _kernel_bundle(output, sources):
    """Publish source snapshots independently from the host binary.

    DeepJIT hashes project includes per entry. Only external SDK identity goes
    into its global signature, so editing one kernel does not invalidate all.
    """
    if not (ROOT / "native/components.json").is_file():
        return {}  # Host-only publication fixtures.
    native = {
        name: digest for name, digest in sources.items() if name.startswith("native/")
    }
    sdk = CANN / "aarch64-linux/asc"
    sdk_files = {
        str(p.relative_to(sdk)): _hash(p)
        for directory in (sdk / "include", sdk / "impl")
        for p in directory.rglob("*")
        if p.is_file()
    }
    signature = _digest(
        {"sdk": sdk_files, "deepjit": _hash(deepjit_root() / "SOURCE.json")}
    )
    identity = _digest({"native": native, "sdk": signature})
    metadata = {"native": native, "sdk_signature": signature}
    identity_headers = {}
    payload_hashes = native

    def validate_snapshot(directory):
        # A directory name is only a claim of identity. Verify copied bytes
        # before publishing, and refuse an existing snapshot that was changed.
        try:
            if directory.is_symlink() or not directory.is_dir():
                raise RuntimeError(f"invalid native source snapshot: {directory}")
            entries = list(directory.rglob("*"))
            if any(path.is_symlink() for path in entries):
                raise RuntimeError(f"symlink in native source snapshot: {directory}")
            actual = {
                str(path.relative_to(directory)) for path in entries if path.is_file()
            }
            expected = set(payload_hashes) | set(identity_headers) | {"bundle.json"}
            if actual != expected:
                raise RuntimeError(
                    f"native source snapshot inventory mismatch: {directory}"
                )
            for relative, digest in payload_hashes.items():
                if _hash(directory / relative) != digest:
                    raise RuntimeError(
                        f"native source snapshot hash mismatch: {directory / relative}"
                    )
            for relative, content in identity_headers.items():
                if (directory / relative).read_text() != content:
                    raise RuntimeError(
                        f"native source snapshot identity mismatch: {directory / relative}"
                    )
            if json.loads((directory / "bundle.json").read_text()) != metadata:
                raise RuntimeError(
                    f"native source snapshot metadata mismatch: {directory}"
                )
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"invalid native source snapshot: {directory}"
            ) from error

    snapshots = output / "sources"
    snapshots.mkdir(exist_ok=True)
    destination = snapshots / identity
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=snapshots))
        try:
            for relative in native:
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            for relative, content in identity_headers.items():
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            _write_json(temporary / "bundle.json", metadata)
            validate_snapshot(temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    else:
        validate_snapshot(destination)
    return {
        "kernel_source_root": str(destination),
        "kernel_signature": signature,
        "kernel_bundle_sha256": identity,
    }


def build_native(
    output=ROOT / "build/native",
    *,
    components=None,
    jobs=2,
    force=False,
):
    """CMake owns dependency tracking, incremental compilation and linking.

    A subset updates the previously selected components. Publication happens
    only after a successful build with unchanged sources; existing processes
    keep their loaded version until restart (Torch registrations are global).
    """
    names = tuple(dict.fromkeys(COMPONENTS if components is None else components))
    if not names or any(name not in COMPONENTS for name in names) or jobs < 1:
        raise ValueError(f"choose components from {COMPONENTS}, jobs >= 1")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    with (output / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(manifest.read_text()) if manifest.exists() else {}
        retained = set(previous.get("components", ()))
        names = sorted(set(names) | (retained & set(COMPONENTS)))
        sources = _sources(names)
        workspace = output / "cmake"
        clean_host = force or _needs_clean_host(workspace, sources)
        host_state = workspace / "verified-inputs.json"
        # A failed build can already have overwritten some objects. Do not
        # compare a later restoration against an older published manifest.
        host_state.unlink(missing_ok=True)
        started = time.monotonic()
        run(
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(workspace),
                f"-DPython3_EXECUTABLE={sys.executable}",
                "-DCMAKE_CXX_COMPILER=g++-13",
                f"-DCANN_ROOT={CANN}",
                f"-DDSV41_DEEPJIT_DIR={deepjit_root()}",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDSV41_COMPONENTS=" + ";".join(names),
            ],
            output / "configure.log",
            started + MAX_BUILD_SECONDS,
        )
        command = ["cmake", "--build", str(workspace), "--parallel", str(jobs)]
        if clean_host:
            command.append("--clean-first")
        run(command, output / "build.log", time.monotonic() + MAX_BUILD_SECONDS)
        bundle = _kernel_bundle(output, sources)
        if _sources(names) != sources:
            raise RuntimeError("sources changed during build; rerun before publishing")
        target = workspace / LIBRARY
        digest = _hash(target)
        _write_json(
            host_state,
            {
                "schema": 1,
                "source_root": str(ROOT.resolve()),
                "sources": sources,
                "library_sha256": digest,
            },
        )
        # A binary can be unchanged after a builder-only edit. Keep its complete
        # provenance immutable too, rather than overwriting the prior manifest.
        release = _digest(
            {
                "host": digest,
                "bundle": bundle,
                "components": names,
                "sources": sources,
            }
        )
        immutable = output / "versions" / release / LIBRARY
        immutable.parent.mkdir(parents=True, exist_ok=True)
        if immutable.exists():
            if _hash(immutable) != digest:
                raise RuntimeError(f"modified immutable extension: {immutable}")
        else:
            temporary = immutable.with_suffix(".tmp")
            shutil.copy2(target, temporary)
            temporary.replace(immutable)
        publication = {
            "schema": 4 if bundle else 3,
            "library": str(immutable),
            "library_sha256": digest,
            "components": names,
            "sources": sources,
            "seconds": time.monotonic() - started,
            **bundle,
        }
        version_manifest = immutable.parent / "manifest.json"
        if version_manifest.exists():
            recorded = json.loads(version_manifest.read_text())

            def identity(value):
                return {k: v for k, v in value.items() if k != "seconds"}

            if identity(recorded) != identity(publication):
                raise RuntimeError(f"modified immutable manifest: {version_manifest}")
            publication = recorded
        else:
            _write_json(version_manifest, publication)
        # Publish complete, loadable versions first. A reader resolving the
        # library alias must always find its matching immutable manifest.
        _write_json(manifest, publication)
        alias = output / LIBRARY
        temporary = alias.with_name(f".{LIBRARY}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(os.path.relpath(immutable, output))
        temporary.replace(alias)
    print(f"Torch-NPU extension: {immutable}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/native")
    parser.add_argument("--components", nargs="+", choices=COMPONENTS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--precompile-device",
        action="store_true",
        help="populate the same DeepJIT cache after the host build",
    )
    parser.add_argument(
        "--precompile-config",
        type=Path,
        help="JSON mapping components to lists of actual geometry arguments",
    )
    parser.add_argument(
        "--physical", type=int, help="physical NPU for explicit precompilation"
    )
    args = parser.parse_args(argv)
    if args.precompile_device and args.physical is None:
        parser.error("--precompile-device requires --physical")
    if args.precompile_config is not None and not args.precompile_device:
        parser.error("--precompile-config requires --precompile-device")
    build_native(
        args.output,
        components=args.components,
        jobs=args.jobs,
        force=args.force,
    )
    if args.precompile_device:
        # Set visibility before importing Torch/Torch-NPU.
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical)
        os.environ["ASCEND_DEVICE_ID"] = "0"
        from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.precompile import (
            precompile,
        )

        precompile(args.output, args.precompile_config)


if __name__ == "__main__":
    main()
