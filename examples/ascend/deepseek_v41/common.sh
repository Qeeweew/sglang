#!/usr/bin/env bash
# Source after setting the profile's physical devices and CPU thread budget.
set -eo pipefail
DSV41_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${DSV41_REPO_ROOT}"
source "${DSV41_CANN_ENV:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}/set_env.sh}"

export ASCEND_DEVICE_ID=0
export SGLANG_DSV41_910B=1
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-1800}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# Native kernels submit directly to ACL; keep Torch submissions on the same path.
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True,per_process_memory_fraction:0.94}"
export PYTHONPATH="${DSV41_REPO_ROOT}/python:${PYTHONPATH:-}"
DSV41_PYTHON="${DSV41_PYTHON:-${DSV41_REPO_ROOT}/build/dsv41-venv/bin/python}"
if ! command -v "${DSV41_PYTHON}" >/dev/null; then
    echo "Missing Python environment: ${DSV41_PYTHON}" >&2
    exit 1
fi
export DSV41_NATIVE_DIR="${DSV41_NATIVE_DIR:-${DSV41_REPO_ROOT}/build/dsv41}"
# Pin lazy operator loads to this immutable build for the process lifetime.
DSV41_NATIVE_LIBRARY=$(readlink -e -- "${DSV41_NATIVE_DIR}/libdsv41_kernels.so") || {
    echo "Missing native library in ${DSV41_NATIVE_DIR}. Build with:" >&2
    echo "${DSV41_PYTHON} -m sglang.srt.hardware_backend.npu.dsv41.kernels.build --backend native --output ${DSV41_NATIVE_DIR} --jobs 2" >&2
    exit 1
}
export DSV41_NATIVE_DIR="$(dirname -- "${DSV41_NATIVE_LIBRARY}")"
# Share device preparation across ranks, outside immutable source publications.
export DSV41_JIT_CACHE_DIR="${DSV41_JIT_CACHE_DIR:-${DSV41_REPO_ROOT}/build/dsv41-jit}"

DSV41_CHECK_ONLY=0
DSV41_ARGS=()
for argument in "$@"; do
    if [[ "${argument}" == --check-only ]]; then
        DSV41_CHECK_ONLY=1
    else
        DSV41_ARGS+=("${argument}")
    fi
done
# Read the final scalar override so Graph coverage and topology checks use
# the same values as SGLang. Remaining arguments are forwarded unchanged.
dsv41_option() {
    local option=$1 value=$2 index argument
    for ((index=0; index<${#DSV41_ARGS[@]}; index++)); do
        argument=${DSV41_ARGS[index]}
        case "${argument}" in
            "${option}")
                value=${DSV41_ARGS[index+1]:-}
                if [[ -z "${value}" || "${value}" == --* ]]; then
                    echo "${option} requires a value" >&2
                    exit 2
                fi
                ;;
            "${option}="*) value=${argument#*=} ;;
        esac
    done
    printf '%s\n' "${value}"
}
dsv41_positive_integer() {
    if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "$1 must be a positive integer" >&2
        exit 2
    fi
}

export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
export SGLANG_DSPARK_USE_GREEDY_DRAFT="${SGLANG_DSPARK_USE_GREEDY_DRAFT:-1}"
DSV41_MODEL_ARGS=(
    --model-path "${DSV41_MODEL_PATH:?Set DSV41_MODEL_PATH to a DeepSeek-V4.1 checkpoint}"
    --device npu --dtype bfloat16 --kv-cache-dtype bfloat16 --page-size 128
    --disable-shared-experts-fusion
    --mm-attention-backend ascend_attn --image-processor-backend pil
    --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41
    --attention-backend dsv4 --enable-cache-report
    --watchdog-timeout "${DSV41_WATCHDOG_TIMEOUT:-1800}"
)
