"""Exercise CMake dependency tracking and transactional library publication."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from sglang.srt.hardware_backend.npu.dsv41.kernels.builders import native as build


def test_subset_identity_tracks_cross_component_header_closure(tmp_path, monkeypatch):
    native = tmp_path / "native"
    for component in ("mhc", "elementwise", "common", "unrelated"):
        (native / component).mkdir(parents=True)
    (tmp_path / "CMakeLists.txt").write_text("# fixture\n")
    (native / "mhc/mhc.asc").write_text('#include "../elementwise/vector.h"\n')
    vector = native / "elementwise/vector.h"
    vector.write_text('#include "nested.h"\n#define VALUE 1\n')
    (native / "common/nested.h").write_text('#include "../elementwise/vector.h"\n')
    (native / "unrelated/other.h").write_text("#define OTHER 3\n")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    before = build._sources(["mhc"])
    assert "native/elementwise/vector.h" in before
    assert "native/common/nested.h" in before
    assert "native/unrelated/other.h" not in before
    vector.write_text('#include "nested.h"\n#define VALUE 2\n')
    after = build._sources(["mhc"])
    assert before.keys() == after.keys()
    assert before["native/elementwise/vector.h"] != after["native/elementwise/vector.h"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    if shutil.which("cmake") is None or shutil.which("g++") is None:
        pytest.skip("CMake and a host C++ compiler are required")
    # Keep reference dependencies under this test's private workspace as well.
    root = tmp_path / "project"
    root.mkdir()
    native = root / "native"
    native.mkdir()
    (native / "value.h").write_text("#define VALUE 7\n")
    (native / "example.cpp").write_text(
        '#include "value.h"\nextern "C" int value() { return VALUE; }\n'
    )
    (root / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.22)
project(publication_test LANGUAGES CXX)
foreach(component IN LISTS DSV41_COMPONENTS)
  list(APPEND sources "native/${component}.cpp")
endforeach()
add_library(dsv41_kernels SHARED ${sources})
""")
    probe = tmp_path / "probe.cpp"
    probe.write_text("""#include <dlfcn.h>
#include <cstdio>
int main(int argc, char **argv) {
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) return 1;
    auto value = reinterpret_cast<int (*)()>(dlsym(lib, argv[2]));
    if (!value) return 2;
    printf("%d", value());
    return 0;
}
""")
    subprocess.run(
        ["g++", str(probe), "-ldl", "-o", str(tmp_path / "probe")], check=True
    )
    vendor = root / "third_party/deepjit"
    vendor.mkdir(parents=True, exist_ok=True)
    (vendor / "SOURCE.json").write_text('{"commit": "fixture"}')
    monkeypatch.setattr(build, "deepjit_root", lambda: vendor)
    monkeypatch.setattr(build, "ROOT", root)
    monkeypatch.setattr(build, "COMPONENTS", ("example", "second", "duplicate"))
    return native, root / "build", tmp_path / "probe"


def value(probe, library, symbol="value"):
    return int(subprocess.check_output([str(probe), str(library), symbol], text=True))


def test_incremental_header_rebuild_and_immutable_publication(project):
    native, output, probe = project
    build.build_native(output, components=["example"])
    first = (output / build.LIBRARY).resolve()
    version_manifest = first.parent / "manifest.json"
    original_manifest = version_manifest.read_bytes()
    manifest_mtime = version_manifest.stat().st_mtime_ns
    target = output / "cmake" / build.LIBRARY
    initial_mtime = target.stat().st_mtime_ns
    build.build_native(output, components=["example"])
    assert target.stat().st_mtime_ns == initial_mtime
    assert (output / build.LIBRARY).resolve() == first
    assert version_manifest.read_bytes() == original_manifest
    assert version_manifest.stat().st_mtime_ns == manifest_mtime
    (native / "value.h").write_text("#define VALUE 19\n")
    build.build_native(output, components=["example"])
    current = (output / build.LIBRARY).resolve()
    assert current != first and first.is_file()
    assert value(probe, first) == 7 and value(probe, current) == 19


@pytest.mark.parametrize("relative", ["value.h", "example.cpp"])
def test_restored_old_timestamp_rebuilds_actual_host_code(project, relative):
    native, output, probe = project
    source = native / relative
    original = source.read_bytes()
    timestamp = source.stat().st_mtime_ns
    build.build_native(output, components=["example"])
    first = (output / build.LIBRARY).resolve()
    replacement = (
        b"#define VALUE 19\n"
        if relative.endswith(".h")
        else b'extern "C" int value() { return 19; }\n'
    )
    for contents, expected in [(replacement, 19), (original, 7)]:
        source.write_bytes(contents)
        os.utime(source, ns=(timestamp, timestamp))
        build.build_native(output, components=["example"])
        current = (output / build.LIBRARY).resolve()
        assert value(probe, current) == expected
        assert value(probe, first) == 7


def test_failed_build_then_timestamp_restoration_cannot_reuse_objects(
    project, monkeypatch
):
    native, output, probe = project
    header = native / "value.h"
    original = header.read_bytes()
    timestamp = header.stat().st_mtime_ns
    build.build_native(output, components=["example"])
    previous = (output / build.LIBRARY).resolve()
    published = (output / "manifest.json").read_bytes()
    header.write_text("#define VALUE 19\n")
    os.utime(header, ns=(timestamp, timestamp))
    bundle = build._kernel_bundle

    def fail_after_host_build(*args):
        assert value(probe, output / "cmake" / build.LIBRARY) == 19
        raise RuntimeError("injected failure after object replacement")

    monkeypatch.setattr(build, "_kernel_bundle", fail_after_host_build)
    with pytest.raises(RuntimeError, match="after object replacement"):
        build.build_native(output, components=["example"])
    assert not (output / "cmake/verified-inputs.json").exists()
    assert (output / "manifest.json").read_bytes() == published
    assert (output / build.LIBRARY).resolve() == previous
    header.write_bytes(original)
    os.utime(header, ns=(timestamp, timestamp))
    monkeypatch.setattr(build, "_kernel_bundle", bundle)
    build.build_native(output, components=["example"])
    assert value(probe, (output / build.LIBRARY).resolve()) == 7


@pytest.mark.parametrize("damage", ["missing-ledger", "external-host-build"])
def test_unverified_cmake_workspace_is_cleaned(project, damage):
    native, output, probe = project
    header = native / "value.h"
    original, timestamp = header.read_bytes(), header.stat().st_mtime_ns
    build.build_native(output, components=["example"])
    header.write_text("#define VALUE 19\n")
    subprocess.run(["cmake", "--build", str(output / "cmake")], check=True)
    assert value(probe, output / "cmake" / build.LIBRARY) == 19
    header.write_bytes(original)
    os.utime(header, ns=(timestamp, timestamp))
    if damage == "missing-ledger":
        (output / "cmake/verified-inputs.json").unlink()
    build.build_native(output, components=["example"])
    assert value(probe, (output / build.LIBRARY).resolve()) == 7


def test_provenance_edit_keeps_previous_version_manifest(project):
    native, output, probe = project
    builders = native.parent / "builders"
    builders.mkdir()
    recipe = builders / "recipe.py"
    recipe.write_text("# original build recipe\n")
    build.build_native(output, components=["example"])
    first = (output / build.LIBRARY).resolve()
    original = (first.parent / "manifest.json").read_bytes()
    recipe.write_text("# updated build recipe; identical host/device code\n")
    build.build_native(output, components=["example"])
    second = (output / build.LIBRARY).resolve()
    assert second != first
    assert first.read_bytes() == second.read_bytes()
    assert (first.parent / "manifest.json").read_bytes() == original
    assert value(probe, first) == value(probe, second) == 7


@pytest.mark.parametrize("failed_manifest", ["version", "current"])
def test_manifest_failure_never_exposes_an_incomplete_version(
    project, monkeypatch, failed_manifest
):
    native, output, probe = project
    build.build_native(output, components=["example"])
    previous = (output / build.LIBRARY).resolve()
    manifest = (output / "manifest.json").read_bytes()
    (native / "value.h").write_text("#define VALUE 29\n")
    write = build._write_json

    def fail_publication(path, publication):
        if path.name == "manifest.json":
            assert (output / build.LIBRARY).resolve() == previous
            if (path.parent == output) == (failed_manifest == "current"):
                raise OSError("injected manifest publication failure")
        write(path, publication)

    monkeypatch.setattr(build, "_write_json", fail_publication)
    with pytest.raises(OSError, match="injected manifest"):
        build.build_native(output, components=["example"])
    assert (output / build.LIBRARY).resolve() == previous
    assert (output / "manifest.json").read_bytes() == manifest
    assert value(probe, previous) == 7
    monkeypatch.setattr(build, "_write_json", write)
    build.build_native(output, components=["example"])
    assert value(probe, (output / build.LIBRARY).resolve()) == 29


def test_partial_build_refreshes_other_header_dependents(project):
    native, output, probe = project
    (native / "second.cpp").write_text(
        '#include "value.h"\nextern "C" int second() { return VALUE + 1; }\n'
    )
    build.build_native(output, components=["example", "second"])
    (native / "value.h").write_text("#define VALUE 23\n")
    build.build_native(output, components=["second"])
    current = (output / build.LIBRARY).resolve()
    assert value(probe, current) == 23 and value(probe, current, "second") == 24
    assert json.loads((output / "manifest.json").read_text())["components"] == [
        "example",
        "second",
    ]


@pytest.mark.parametrize("failure", ["compile", "link"])
def test_failure_preserves_published_library_and_manifest(project, failure):
    native, output, probe = project
    build.build_native(output, components=["example"])
    previous = (output / build.LIBRARY).resolve()
    manifest = (output / "manifest.json").read_bytes()
    if failure == "compile":
        (native / "example.cpp").write_text("invalid C++\n")
        selection = ["example"]
    else:
        (native / "duplicate.cpp").write_text('extern "C" int value() { return 99; }\n')
        selection = ["duplicate"]
    with pytest.raises(RuntimeError, match="build failed"):
        build.build_native(output, components=selection)
    assert (output / build.LIBRARY).resolve() == previous
    assert (output / "manifest.json").read_bytes() == manifest
    assert value(probe, previous) == 7
    assert (output / "build.log").read_text()


def test_device_edit_publishes_source_snapshot_without_relink(project, monkeypatch):
    """A live host binary survives device edits; each release keeps its sources."""
    native, output, probe = project
    (native / "components.json").write_text(
        '{"components": [{"name": "example", "backend": "jit"}]}'
    )
    (native / "example").mkdir()
    kernel = native / "example/kernel.asc"
    kernel.write_text("// device version one\n")
    vendor = native.parent / "third_party/deepjit"
    vendor.mkdir(parents=True, exist_ok=True)
    (vendor / "SOURCE.json").write_text('{"commit": "fixture"}')
    monkeypatch.setattr(build, "CANN", native.parent / "fake-cann")
    build.build_native(output, components=["example"])
    first = json.loads((output / "manifest.json").read_text())
    original_mtime = (output / "cmake" / build.LIBRARY).stat().st_mtime_ns
    original_snapshot = first["kernel_source_root"]
    kernel.write_text("// device version two\n")
    build.build_native(output, components=["example"])
    second = json.loads((output / "manifest.json").read_text())
    assert (output / "cmake" / build.LIBRARY).stat().st_mtime_ns == original_mtime
    assert first["library_sha256"] == second["library_sha256"]
    assert first["library"] != second["library"]
    assert first["kernel_source_root"] != second["kernel_source_root"]
    assert (
        Path(original_snapshot) / "native/example/kernel.asc"
    ).read_text() == "// device version one\n"
    assert value(probe, first["library"]) == value(probe, second["library"]) == 7


@pytest.fixture
def kernel_project(project, monkeypatch):
    native, output, probe = project
    (native / "components.json").write_text(
        '{"components": [{"name": "example", "backend": "jit"}]}'
    )
    kernel = native / "example/kernel.asc"
    kernel.parent.mkdir()
    kernel.write_text("// device version one\n")
    vendor = native.parent / "third_party/deepjit"
    vendor.mkdir(parents=True, exist_ok=True)
    (vendor / "SOURCE.json").write_text('{"commit": "fixture"}')
    monkeypatch.setattr(build, "CANN", native.parent / "fake-cann")
    return native, output, probe, kernel, vendor


@pytest.mark.parametrize("dependency", ["device"])
def test_source_copy_race_preserves_publication_and_can_retry(
    kernel_project, monkeypatch, dependency
):
    native, output, probe, kernel, vendor = kernel_project
    build.build_native(output, components=["example"])
    previous = (output / build.LIBRARY).resolve()
    manifest = (output / "manifest.json").read_bytes()
    previous_snapshot = Path(json.loads(manifest)["kernel_source_root"])
    snapshots_before = set((output / "sources").iterdir())

    source = kernel
    original = source.read_bytes()
    candidate = original.replace(b"one", b"two")
    source.write_bytes(candidate)
    copy = build.shutil.copy2
    injected = False

    def change_during_copy(first, second, *args, **kwargs):
        nonlocal injected
        if Path(first) == source and not injected:
            # Change the real dependency after its identity was selected but
            # before copying its bytes, reproducing the publication race.
            source.write_bytes(b"// changed during snapshot copy\n")
            injected = True
        return copy(first, second, *args, **kwargs)

    monkeypatch.setattr(build.shutil, "copy2", change_during_copy)
    with pytest.raises(RuntimeError, match="source snapshot hash mismatch"):
        build.build_native(output, components=["example"])
    assert injected
    assert (output / build.LIBRARY).resolve() == previous
    assert (output / "manifest.json").read_bytes() == manifest
    assert set((output / "sources").iterdir()) == snapshots_before
    assert (previous_snapshot / "native/example/kernel.asc").read_bytes() == (
        b"// device version one\n"
    )
    assert value(probe, previous) == 7

    # Restoring the intended input must not reuse a poisoned old-identity path.
    monkeypatch.setattr(build.shutil, "copy2", copy)
    source.write_bytes(candidate)
    build.build_native(output, components=["example"])
    current = json.loads((output / "manifest.json").read_text())
    snapshot = Path(current["kernel_source_root"])
    relative = "native/example/kernel.asc"
    assert snapshot != previous_snapshot
    assert (snapshot / relative).read_bytes() == candidate
    metadata = json.loads((snapshot / "bundle.json").read_text())
    recorded = metadata["native"][relative]
    assert build._hash(snapshot / relative) == recorded
    assert value(probe, current["library"]) == 7


@pytest.mark.parametrize(
    "relative, expected_error",
    [
        ("native/example/kernel.asc", "hash"),
        ("bundle.json", "metadata"),
        ("native/example/untracked.h", "inventory"),
    ],
)
def test_existing_polluted_source_snapshot_is_rejected(
    kernel_project, relative, expected_error
):
    _, output, probe, _, _ = kernel_project
    build.build_native(output, components=["example"])
    previous = (output / build.LIBRARY).resolve()
    manifest = (output / "manifest.json").read_bytes()
    snapshot = Path(json.loads(manifest)["kernel_source_root"])
    polluted = snapshot / relative
    polluted.write_text("{}\n" if relative == "bundle.json" else "// polluted\n")

    for _ in range(2):
        with pytest.raises(
            RuntimeError, match=f"source snapshot {expected_error} mismatch"
        ):
            build.build_native(output, components=["example"])
        assert (output / build.LIBRARY).resolve() == previous
        assert (output / "manifest.json").read_bytes() == manifest
        assert value(probe, previous) == 7
        assert not list((output / "sources").glob(".publish-*"))


def test_deepjit_header_changes_keep_provenance_immutable(project):
    native, output, _ = project
    vendor = build.deepjit_root()
    header = vendor / "header.hpp"
    header.write_text("// vendor one\n")
    build.build_native(output, components=["example"])
    first = (output / build.LIBRARY).resolve()
    before = (first.parent / "manifest.json").read_bytes()
    header.write_text("// vendor two\n")
    build.build_native(output, components=["example"])
    second = (output / build.LIBRARY).resolve()
    assert first != second
    assert (first.parent / "manifest.json").read_bytes() == before
    after = json.loads((second.parent / "manifest.json").read_text())
    assert after["sources"]["3rdparty/deepjit/header.hpp"] == build._hash(header)
