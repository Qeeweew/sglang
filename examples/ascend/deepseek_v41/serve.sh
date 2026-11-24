#!/usr/bin/env bash
# Eight physical NPUs: attention DP8/TP1 or DP4/TP2, MoE EP8 and DSpark.
set -eo pipefail
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"

export SGLANG_SHARED_EXPERT_TP1=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1
export SGLANG_DP_USE_REDUCE_SCATTER="${SGLANG_DP_USE_REDUCE_SCATTER:-1}"
# This SGLang build's variable-length collectives require PyNCCL.
export SGLANG_DP_USE_GATHERV=0
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
DSV41_ATTN_DP_SIZE=$(dsv41_option --attn-dp-size "${DSV41_ATTN_DP_SIZE:-4}")
if [[ "${DSV41_ATTN_DP_SIZE}" != 4 && "${DSV41_ATTN_DP_SIZE}" != 8 ]]; then
    echo "Attention DP size must be 4 or 8" >&2
    exit 2
fi
if [[ "$(dsv41_option --tp-size 8)" != 8 || "$(dsv41_option --ep-size 8)" != 8 ]]; then
    echo "The eight-card profile requires --tp-size 8 and --ep-size 8" >&2
    exit 2
fi
export DSV41_ATTN_DP_SIZE
DSV41_KV_CAP_ARGS=()
if [[ -n "${DSV41_MAX_TOTAL_TOKENS:-}" ]]; then
    dsv41_positive_integer max-total-tokens "${DSV41_MAX_TOTAL_TOKENS}"
    DSV41_KV_CAP_ARGS+=(--max-total-tokens "${DSV41_MAX_TOTAL_TOKENS}")
fi
DSV41_REQUESTS=$(dsv41_option --max-running-requests "${DSV41_MAX_RUNNING_REQUESTS:-64}")
dsv41_positive_integer max-running-requests "${DSV41_REQUESTS}"
DSV41_LOCAL_REQUESTS=$(((DSV41_REQUESTS + DSV41_ATTN_DP_SIZE - 1) / DSV41_ATTN_DP_SIZE))
DSV41_GRAPH_BATCHES=()
for ((batch=1; batch<=DSV41_LOCAL_REQUESTS; batch++)); do
    DSV41_GRAPH_BATCHES+=("${batch}")
done
# DSpark inherits refill batching: at local capacity 8 it otherwise waits for
# two free slots and can strand the eighth request. Trailing user args win.
if [[ "${DSV41_CHECK_ONLY}" == 1 ]]; then
    "${DSV41_PYTHON}" examples/ascend/deepseek_v41/check.py
    exit 0
fi
"${DSV41_PYTHON}" examples/ascend/deepseek_v41/check.py --quiet
exec "${DSV41_PYTHON}" -m sglang.cli.main serve \
    "${DSV41_MODEL_ARGS[@]}" \
    --tp-size 8 --attn-dp-size "${DSV41_ATTN_DP_SIZE}" --dp-size 1 --ep-size 8 \
    --moe-a2a-backend none --enable-dp-lm-head --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
    --enable-decoder-swa-bounded-replay --enable-encoder-swa-bounded-replay \
    --cuda-graph-bs-decode "${DSV41_GRAPH_BATCHES[@]}" \
    --max-running-requests "${DSV41_REQUESTS}" \
    --min-free-slots-delay 1 \
    --chunked-prefill-size "${DSV41_PREFILL_CHUNK_SIZE:-32768}" \
    "${DSV41_KV_CAP_ARGS[@]}" \
    --context-length "${DSV41_CONTEXT_LENGTH:-1048576}" \
    --mem-fraction-static "${DSV41_MEM_FRACTION_STATIC:-0.88}" \
    --host 127.0.0.1 --port "${DSV41_PORT:-31088}" "${DSV41_ARGS[@]}"
