# Ascend 910B V4.1 qualification

Run from the SGLang root with a clean Python environment and the selected CANN
SDK. Unit tests cover checkpoint pairing, replay inputs, scratch ownership,
host build publication, toolchain provenance, DeepJIT process locks, and PCH.
Device scripts use independent numerical references and exercise Graph replay.

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONPATH="$PWD/python:$PWD/test/ascend:${PYTHONPATH:-}"
export SGLANG_DSV41_910B=1
export DSV41_PYTHON="$PWD/build/dsv41-venv/bin/python"
"$DSV41_PYTHON" -m pytest test/ascend/dsv41/kernels/unit \
  test/ascend/dsv41/test_dsv41_weights.py \
  test/ascend/dsv41/test_dsv41_expert_scratch.py \
  test/ascend/dsv41/test_dsv41_replay_inputs.py \
  test/ascend/dsv41/test_checkpoint_progress.py \
  test/ascend/dsv41/test_native_integration.py -q
```

Device tests require the built native publication. Bind an idle physical NPU
before importing Torch. The following checks use physical NPU7, mapped to 0:

```bash
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_DEVICE_ID=0 TASK_QUEUE_ENABLE=0
export DSV41_NATIVE_DIR="$PWD/build/dsv41"
export DSV41_JIT_CACHE_DIR="$PWD/build/dsv41-jit"
"$DSV41_PYTHON" -m dsv41.kernels.operators.test_router --library-dir "$DSV41_NATIVE_DIR"
"$DSV41_PYTHON" -m dsv41.kernels.operators.test_ced --library-dir "$DSV41_NATIVE_DIR"
"$DSV41_PYTHON" -m dsv41.kernels.operators.test_w8a16 \
  --physical 7 --output build/checks/w8a16
"$DSV41_PYTHON" -m dsv41.kernels.operators.test_route_bucket \
  --physical 7 --output build/checks/route-bucket.json
"$DSV41_PYTHON" -m dsv41.kernels.operators.test_router_dp_graph \
  --physical 7 --output build/checks/router-dp-graph
"$DSV41_PYTHON" -m dsv41.kernels.integration.test_local_bucket_moe \
  --physical 7 --output build/checks/local-bucket-moe.json
SGLANG_DSV41_TEST_NPU=1 "$DSV41_PYTHON" -m pytest -q \
  test/ascend/dsv41/test_dsv41_w8a16.py
```

Other numerical scripts live in `kernels/operators`. Run them as modules and
use `--help` to see their geometry and output options. Use pytest for the
unit and adapter paths shown above. Keep logs, JSON, traces, and binary
artifacts under ignored `build/`.

For a dedicated eight-card server started with the example profile, run:

```bash
"$DSV41_PYTHON" examples/ascend/deepseek_v41/smoke.py \
  --url http://127.0.0.1:31088 --parallel-requests 8 --prefix-replay \
  --output build/checks/dsv41-http.json
```

This checks finite generation logprobs for each request and a page-sized or
larger cache hit on the repeated long prompt. It does not assert batch-invariant
outputs or equality between approximate bounded replay and full prefill.
