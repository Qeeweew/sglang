#!/usr/bin/env bash
# Reuse the container's NPU wheels without its editable SGLang or .pth hooks.
set -euo pipefail
DSV41_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
DSV41_BASE_PYTHON=${DSV41_BASE_PYTHON:-python3}
DSV41_VENV=${DSV41_VENV:-${DSV41_REPO_ROOT}/build/dsv41-venv}
"${DSV41_BASE_PYTHON}" -m venv "${DSV41_VENV}"
DSV41_CONTAINER_SITE=$("${DSV41_BASE_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
DSV41_CONTAINER_SITE="${DSV41_CONTAINER_SITE}" DSV41_SOURCE="${DSV41_REPO_ROOT}/python" \
    "${DSV41_VENV}/bin/python" - <<'PY'
import os
import sysconfig
from pathlib import Path
# Plain paths expose wheels but deliberately do not execute their .pth files.
Path(sysconfig.get_path("purelib"), "dsv41-container-wheels.pth").write_text(
    os.environ["DSV41_CONTAINER_SITE"] + "\n" + os.environ["DSV41_SOURCE"] + "\n"
)
PY
"${DSV41_VENV}/bin/python" -m pip install blake3 deepseek-recipe==0.2.0
printf 'DSV41_PYTHON=%s/bin/python\n' "${DSV41_VENV}"
