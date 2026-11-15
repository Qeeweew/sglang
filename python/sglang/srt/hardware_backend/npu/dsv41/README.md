# DeepSeek V4.1 on Ascend 910B

This backend serves DeepSeek V4.1 on Ascend 910B. Routed MXFP4 experts remain
compressed on the NPUs, and Engram uses registered compressed host tables.

The current target is Ascend 910B3, CANN 9.1, and a Torch/Torch-NPU installation
with matching ABIs. Native host compilation requires GCC 13+, CMake 3.22+,
CANN development headers, and libnuma (used for registered Engram memory).
The checkpoint must have FP8 dense weights with 32x32 E8M0 scales and MXFP4
routed experts.

## Layout

| Path | Responsibility |
| --- | --- |
| `attention/` | Attention, Q/K projections, BF16 KV and INT8 Indexer pools |
| `prefill/` | CED prefill and bounded SWA replay |
| `dspark/` | Draft attention, KV injection and target verification |
| `moe.py`, `parallel.py` | Resident MXFP4 experts, compact scratch and DP collectives |
| `checkpoint.py`, `dense_weights.py`, `linear.py`, `engram.py` | Checkpoint loading and persistent weight layouts |
| `kernels/ops` | Python operator contracts and prepared invocations |
| `kernels/native` | Ascend C sources, C++ bindings, and launch plans |
| `kernels/runtime` | Library loading and explicit device preparation |
| `kernels/builders` | Incremental host builds and immutable publication |
| repository `3rdparty/deepjit` | Pinned compiler/cache/loader dependency |

The adapters are wired at SGLang definitions and factories. There is no global
patch installer or runtime Python source rewriting. Set `SGLANG_DSV41_910B=1`
before importing SGLang to select this implementation. Other launches retain
their existing implementations.

## Prepare the container environment

Run these commands from the SGLang repository root:

```bash
bash examples/ascend/deepseek_v41/prepare_env.sh
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export DSV41_PYTHON="$PWD/build/dsv41-venv/bin/python"
"$DSV41_PYTHON" -m sglang.srt.hardware_backend.npu.dsv41.kernels.build \
  --output "$PWD/build/dsv41" --jobs 2
```

The environment script exposes the installed NPU wheels and installs `blake3`
and the V4.1 `deepseek-recipe==0.2.0` prompt/parser dependency. Set
`DSV41_BASE_PYTHON` or `DSV41_VENV` to select another base interpreter or
environment directory. `DSV41_PYTHON` can select an existing environment.
The launch check verifies that SGLang imports from this checkout.

## Serve on eight NPUs

```bash
export DSV41_MODEL_PATH=/path/to/DeepSeek-V4.1-checkpoint
bash examples/ascend/deepseek_v41/serve.sh --check-only
bash examples/ascend/deepseek_v41/serve.sh
# For attention DP4/TP2
DSV41_ATTN_DP_SIZE=4 bash examples/ascend/deepseek_v41/serve.sh
```

Both profiles use MoE EP8, separate shared experts, DSpark, CED, encoder and
decoder SWA replay, and target/draft NPU Graph. Attention DP8/TP1 uses a 4,096-token
local prefill chunk. Attention DP4/TP2 uses 8,192. Both default to a 1,048,576-token
context limit, 64 global running requests, automatic KV sizing, and memory fraction 0.88.

For a smaller HTTP smoke check, the context and request budgets can be overridden:

```bash
DSV41_CONTEXT_LENGTH=8192 DSV41_MAX_TOTAL_TOKENS=8192 \
DSV41_MAX_RUNNING_REQUESTS=8 DSV41_PREFILL_CHUNK_SIZE=4096 \
  bash examples/ascend/deepseek_v41/serve.sh
"$DSV41_PYTHON" examples/ascend/deepseek_v41/smoke.py
```

Add `--parallel-requests 8 --prefix-replay --output build/checks/dsv41-http.json`
to the smoke command to check concurrent generation and a repeated long prompt.

Weight loading uses SGLang's progress display on rank 0, with separate target
and draft bars that advance as tensors are consumed. Normal logs keep model
and memory summaries; `--log-level debug` also shows per-layer expert storage,
Engram backing details, and checkpoint read timings.

Target/draft dense and shared expert weights use W8A16 G256. Checkpoint FP8
weights are converted to INT8 at load time, then repacked with raw FP16
scales into N16 layout. Forward consumes and returns BF16 without quantizing
activations. Expert weights stay MXFP4 on device.

Attention and C1/C2 KV use BF16. Indexer keys use INT8 with FP16 scales.
Engram tables retain FP8/E8M0 storage on the host. The resident MoE route
uses Vector TopK followed by one local bucket kernel with INT32 metadata.

Large eager prefill batches compact scratch to the local EP routing count.
Bounded replay uses the model's approximate SWA recovery.

## Build and deployment boundaries

The native builder publishes `libdsv41_kernels.so`, its manifest, and source
snapshots under immutable version directories. Keep the whole publication,
not just the shared library. `DSV41_NATIVE_DIR` chooses a publication;
`DSV41_JIT_CACHE_DIR` chooses a cache shared by the ranks on the node.

Native ABI 22 requires rebuilding the host publication. CMake builds the host
extension. The router produces grouped GEMM metadata on the device.
AICore kernels compile through DeepJIT for the model geometries actually bound.
Prepare and warm them before Graph capture; hot calls and replay do not compile.
Warm eager prefill and encoder replay geometries as well before measuring
performance; their first use may prepare additional JIT entries.
A cold cache needs the CANN compiler. Rebuild when Torch ABI, bindings, SDK, or
vendored headers change, and restart the processes after rebuilding.

Wheel/sdist build hooks copy DeepJIT into package data, including its license
and source identity. Editable checkouts use the single root `3rdparty` tree.
Native source and CMake files are package data; host binaries are built for the
deployment environment and are not committed.

Validation commands and current coverage are documented in
[`test/ascend/dsv41/README.md`](../../../../../../test/ascend/dsv41/README.md).
