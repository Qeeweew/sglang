"""Package/toolchain locations; importing this module has no device effects."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANN = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0"))


def deepjit_root():
    """Find the checked-in headers or the copy packaged in a wheel."""
    packaged = ROOT / "third_party/deepjit"
    if packaged.is_dir():
        return packaged
    for parent in ROOT.parents:
        candidate = parent / "3rdparty/deepjit"
        if (candidate / "SOURCE.json").is_file():
            return candidate
    raise RuntimeError(
        "DeepJIT headers missing; use the SGLang source checkout or rebuild the wheel"
    )


def native_library_path(library=None, *, directory=None):
    if library is None:
        root = Path(
            directory or os.environ.get("DSV41_NATIVE_DIR", ROOT / "build/native")
        )
        library = root / "libdsv41_kernels.so"
    return Path(library).resolve(strict=True)
