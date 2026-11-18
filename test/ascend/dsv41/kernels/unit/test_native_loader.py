"""Torch registrations are process-global; reject duplicate versions safely."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime import native


def test_loader_pins_one_resolved_version(tmp_path, monkeypatch):
    old, new, alias = (tmp_path / name for name in ("old.so", "new.so", "selected.so"))
    old.touch()
    new.touch()
    alias.symlink_to(old)
    loads = []
    source = str(tmp_path / "sources")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "kernel_source_root": source,
                "kernel_signature": "test-source-signature",
            }
        )
    )
    configure = Mock()
    validate = Mock()
    ops = SimpleNamespace(abi=lambda: 25, configure_jit=configure)
    monkeypatch.setattr(native, "_loaded_path", None)
    monkeypatch.setattr(native, "validate_jit_toolchain", validate)
    monkeypatch.setattr(native.torch.ops, "load_library", loads.append)
    monkeypatch.setattr(native.torch.ops, "dsv41", ops)
    assert native.load_library(alias) is ops
    assert native.load_library(alias) is ops
    alias.unlink()
    alias.symlink_to(new)
    assert native.load_library() is ops
    with pytest.raises(RuntimeError, match="restart"):
        native.load_library(alias)
    assert loads == [str(old)]
    configure.assert_called_once()
    validate.assert_called_once()
    assert configure.call_args.args[:2] == (source, "test-source-signature")


def test_failed_load_does_not_publish_a_version(tmp_path, monkeypatch):
    library = tmp_path / "invalid.so"
    library.touch()
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(native, "_loaded_path", None)
    monkeypatch.setattr(native, "validate_jit_toolchain", lambda publication: None)

    def fail(path):
        raise OSError("load failed")

    monkeypatch.setattr(native.torch.ops, "load_library", fail)
    with pytest.raises(OSError, match="load failed"):
        native.load_library(library)
    assert native._loaded_path is None


@pytest.mark.parametrize("abi", [12, 13, 14, 15, 16, 17, 18, 19, 23, 24])
def test_obsolete_native_abi_is_rejected(tmp_path, monkeypatch, abi):
    library = tmp_path / f"abi{abi}.so"
    library.touch()
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(native, "_loaded_path", None)
    monkeypatch.setattr(native, "validate_jit_toolchain", lambda publication: None)
    monkeypatch.setattr(native.torch.ops, "load_library", lambda path: None)
    monkeypatch.setattr(native.torch.ops, "dsv41", SimpleNamespace(abi=lambda: abi))
    with pytest.raises(RuntimeError, match="incompatible.*rebuild"):
        native.load_library(library)
    assert native._loaded_path is None


def test_wrong_sdk_is_rejected_before_torch_registration(tmp_path, monkeypatch):
    library = tmp_path / "candidate.so"
    library.touch()
    (tmp_path / "manifest.json").write_text("{}")
    load = Mock()
    monkeypatch.setattr(native, "_loaded_path", None)
    monkeypatch.setattr(native.torch.ops, "load_library", load)
    monkeypatch.setattr(
        native, "validate_jit_toolchain", Mock(side_effect=RuntimeError("SDK changed"))
    )
    with pytest.raises(RuntimeError, match="SDK changed"):
        native.load_library(library)
    load.assert_not_called()
    assert native._loaded_path is None
