"""Cold-path validation of the SDK that DeepJIT will actually select."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_FALLBACKS = (
    Path("/usr/local/Ascend/ascend-toolkit/latest"),
    Path("/usr/local/Ascend/cann"),
)
_DEEPJIT_SOURCE = "3rdparty/deepjit/SOURCE.json"


def selected_toolkit():
    """Match DeepJIT Ascend::find_ascend_toolkit's selection order."""
    explicit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if explicit:
        root = Path(explicit)
    else:
        root = next((path for path in _FALLBACKS if path.exists()), None)
    if root is None or not root.is_dir():
        raise RuntimeError("Ascend toolkit home was not found")
    return root.resolve()


def sdk_signature(toolkit, deepjit_source_sha256):
    """Use the same SDK-header/vendor ledger as the source-bundle builder."""
    sdk = Path(toolkit) / "aarch64-linux/asc"
    files = {}
    for directory in (sdk / "include", sdk / "impl"):
        if not directory.is_dir():
            raise RuntimeError(f"missing Ascend SDK headers: {directory}")
        for path in directory.rglob("*"):
            if path.is_file():
                files[str(path.relative_to(sdk))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    ledger = {"sdk": files, "deepjit": deepjit_source_sha256}
    return hashlib.sha256(json.dumps(ledger, sort_keys=True).encode()).hexdigest()


def validate_jit_toolchain(publication):
    """Reject a stale SDK signature before registering a native extension.

    This runs once per process/library, before prepare or capture. Warm launches
    neither walk headers nor read environment variables for this validation.
    """
    try:
        vendor = publication["sources"][_DEEPJIT_SOURCE]
        expected = publication["kernel_signature"]
    except (KeyError, TypeError):
        raise RuntimeError(
            "incompatible native publication: missing DeepJIT provenance; rebuild"
        ) from None
    toolkit = selected_toolkit()
    actual = sdk_signature(toolkit, vendor)
    if actual != expected:
        raise RuntimeError(
            f"Ascend SDK headers at {toolkit} differ from this native publication; "
            "rebuild with the selected SDK before preparing kernels"
        )
