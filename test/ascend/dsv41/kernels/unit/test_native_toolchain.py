"""SDK contents, not compiler-version text, bind a DeepJIT publication."""

import pytest
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime import toolchain


def sdk(tmp_path, name):
    root = tmp_path / name
    for directory in ("include", "impl"):
        path = root / "aarch64-linux/asc" / directory
        path.mkdir(parents=True)
        (path / "contract.h").write_text("// fixed SDK contract\n")
    return root


def test_sdk_contents_are_independent_of_installation_path(tmp_path):
    first, second = sdk(tmp_path, "one"), sdk(tmp_path, "two")
    assert toolchain.sdk_signature(first, "vendor") == toolchain.sdk_signature(
        second, "vendor"
    )


def test_changed_sdk_with_same_compiler_version_is_rejected(tmp_path, monkeypatch):
    first, second = sdk(tmp_path, "one"), sdk(tmp_path, "two")
    for root in (first, second):
        (root / "compiler-version.txt").write_text("identical version")
    publication = dict(
        sources={"3rdparty/deepjit/SOURCE.json": "vendor"},
        kernel_signature=toolchain.sdk_signature(first, "vendor"),
    )
    monkeypatch.setenv("ASCEND_HOME_PATH", str(second))
    toolchain.validate_jit_toolchain(publication)
    (second / "aarch64-linux/asc/impl/contract.h").write_text(
        "// changed implementation\n"
    )
    with pytest.raises(RuntimeError, match="headers.*differ.*rebuild"):
        toolchain.validate_jit_toolchain(publication)


def test_sdk_selection_matches_deepjit_environment_order(tmp_path, monkeypatch):
    first, second = sdk(tmp_path, "one"), sdk(tmp_path, "two")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(first))
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(second))
    assert toolchain.selected_toolkit() == first.resolve()
    monkeypatch.delenv("ASCEND_HOME_PATH")
    assert toolchain.selected_toolkit() == second.resolve()
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME")
    monkeypatch.setattr(toolchain, "_FALLBACKS", (tmp_path / "absent", first))
    assert toolchain.selected_toolkit() == first.resolve()


def test_missing_sdk_headers_are_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="missing Ascend SDK headers"):
        toolchain.sdk_signature(tmp_path, "vendor")


def test_legacy_publication_gets_a_rebuild_error():
    with pytest.raises(RuntimeError, match="incompatible.*rebuild"):
        toolchain.validate_jit_toolchain({})
