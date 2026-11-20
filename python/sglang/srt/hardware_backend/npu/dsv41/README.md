# DeepSeek V4.1 on Ascend 910B

This backend serves DeepSeek V4.1 on Ascend 910B. Routed MXFP4 experts remain
compressed on the NPUs, and Engram uses registered compressed host tables.
See the [910B deployment guide](../../../../../../examples/ascend/deepseek_v41/README.md)
for the Docker build, topology selection, and benchmarks.

The current target is Ascend 910B3, CANN 9.1, and a Torch/Torch-NPU installation
with matching ABIs. Native host compilation requires GCC 13+, CMake 3.22+,
CANN development headers, and libnuma (used for registered Engram memory).
The source checkpoint has FP8 dense weights with 32x32 E8M0 scales and MXFP4
routed experts. The backend also recognizes the target-only
[`quarot_h32_int4_g32_v1` export](../../../../../../examples/ascend/deepseek_v41/quarot-g32.md).

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

With the source checkpoint, target/draft dense and shared expert weights use W8A16 G256. Checkpoint FP8
weights are converted to INT8 at load time, then repacked with raw FP16
scales into N16 layout. Dense layers and the model boundary stay BF16. For the
legacy target's TP1 shared expert, EP dispatch and the router still consume
BF16. After routing, one local gather writes both the owned shared inputs and
the expert-major routed inputs as FP16. Both gate/up projections output FP16;
one clipped SwiGLU computes both in FP32 and stores FP16 directly into the two
down inputs. Shared W8A16 down writes BF16 directly from its FP32 accumulator,
whereas routed MXFP4 down writes FP16 for FP32 weighted combine. The peer wire
format is unchanged. Source MXFP4 and G32 target experts use the same EP8
schedule: transport does not run shared before gather. Fixed-size shards and
unequal DP segments both pass their owner rows to the joint SwiGLU, and add
shared only after the routed sum returns. Source MXFP4 experts use FP16
activations for both target and draft projections.

The G32 checkpoint instead selects signed INT4 × INT8 grouped GEMM for the
target experts without any environment switch. Both shared and routed target
projections use fused FP32 block-Hadamard + per-token A8. Shared weights use
per-output-channel W8 and native Torch-NPU `npu_quant_matmul`. Draft/MTP is
unmodified. G32 routed weight reads use the same L2 bypass as MXFP4. Routed
FP16 output uses full FP32 channel and token scales.

W8A16 uses two Vector lanes per Cube. Each lane loads INT8 and FP16 scales,
then writes the decoded panel to a double-buffered GM workspace. Cube imports
it into L1 before reusing the slot. Every row tile decodes the weight again;
the workspace round trip alone moves four logical bytes per weight element
per row tile. These bytes need not reach HBM if the workspace hits in L2.
For a single row tile, automatic planning compares the weight work and
number of K panels on the busiest Cube, rather than requiring a full first
wave. Explicit plans and multi-row-tile prefill keep the previous selection.
Native prepare selects and validates the tile, allocates missing output, and
prepares the JIT entry before Graph capture. Launch uses the returned tile and
requests scratch for the active Cube workers, retaining both slots per worker.

W8A16, grouped GEMM, attention, QLI, mHC projection, router bucketing and
DSpark schedule allocate private scratch in the C++ launch on the current
stream. Prepare does not reserve these buffers. `raw_alloc_with_stream` uses
the same Torch-NPU caching allocator as tensor allocations without constructing
a temporary tensor. Sequential operators can reuse its storage; during capture
the graph pool owns the recorded addresses. Replay does not allocate scratch.
Explicit buffers, where supported, are caller-owned and must survive captured use.
Do not overlap replays that alias a shared graph pool. Outputs and state that
cross operator boundaries (including routing metadata, peer communication and
KV) retain their existing ownership and ordering contracts.

Attention and C1/C2 KV use BF16. Indexer keys use INT8 with FP16 scales.
Engram tables retain FP8/E8M0 storage on the host. The resident MoE route
uses Vector TopK followed by one local bucket kernel with INT32 metadata.
Grouped GEMM bypasses L2 for packed MXFP4 weight reads in decode and prefill.
The FP16 handoff between AIV and AIC stays cached. This avoids filling L2
with streamed weights but can increase HBM reads when weights would otherwise
be reused from L2 across M tiles or calls.

Large eager prefill batches compact scratch to the local EP routing count.
Bounded replay uses the model's approximate SWA recovery.

## Native operator library

The component name in `kernels/native/components.json` selects a directory for
the host build and the DeepJIT source snapshot. The Python operator wrappers
live in `kernels/ops/`. In each native component, `binding.cpp` checks tensor
contracts, `launch.cpp` selects a prepared kernel on the current stream, and
the `.asc` entry runs the device pipeline.

| Computation | Native components |
| --- | --- |
| Dense and expert projections | `projection`, `w8a16`, `quant_grouped_gemm` (MXFP4 weights × FP16), `int4_g32_a8` (signed G32 weights × INT8) |
| Format conversion | `fp8_weight`, `mxfp4_weight`, `h32_quant` (H32 transform and token quantization for both shared and routed projections) |
| Attention, KV and index selection | `attention`, `qli`, `position_quant`, `compressor`, `page_metadata`, `ced` |
| Routing, activation and model-specific operations | `router`, `moe_routes`, `elementwise`, `mhc`, `engram`, `engram_ops` |
| Speculative decoding and communication | `dspark_schedule`, `dspark_markov`, `peer_all_reduce` |

`common/grouped_schedule.h` maps compact route records to the same sequence of
tiles on the Cube and both Vector lanes for either expert weight format.
`common/cube.h` owns the 910B short-MMAD barrier and the 16-bit L1 operand
imports. `common/device.h` supplies typed UB byte offsets and immediate
same-core event handoffs. Each pipeline assigns its own event IDs and owns its
split-phase and cross-core synchronization. On the host, `common/actual_entry.h`
caches geometry-specialized JIT handles per thread.

## Build and deployment boundaries

The native builder publishes `libdsv41_kernels.so`, its manifest, and source
snapshots under immutable version directories. Keep the whole publication,
not just the shared library. `DSV41_NATIVE_DIR` chooses a publication;
`DSV41_JIT_CACHE_DIR` chooses a cache shared by the ranks on the node.

Native ABI 29 requires rebuilding the host publication. CMake builds the host
extension. The router produces grouped GEMM metadata on the device.
AICore kernels compile through DeepJIT for the model geometries actually bound.
Prepare and warm them before Graph capture; hot calls and replay do not compile.
Warm eager prefill and encoder replay geometries as well before measuring
performance; their first use may prepare additional JIT entries.
A cold cache needs the CANN compiler. Rebuild when Torch ABI, bindings, SDK, or
vendored headers change, and restart the processes after rebuilding.

`AscendMarkovStep` uses the optional `dspark_markov` component on either side
of the existing candidate all-gather. For greedy FP32 steps without saved
logits the local kernel reads row-strided BF16/FP32 logits and bias, adds in
FP32 and selects the first winning token in each vocabulary tile. Wide shards
reduce tile winners on the same device before exchanging the usual single
`(score, bitcast token ID)` per row. The final rank merge emits INT64 tokens.
There is no TP-specific path. Sampled steps, saved logits and unsupported
types use the original Torch local selection; strided gathered packets use
the original Torch merge. Compatible publications fall back for each missing
optional operator. Use the ordinary eager draft warmup or explicitly prepare with
`torch.ops.dsv41.precompile_dspark_markov()` on the initialized current NPU
before standalone Graph capture.

Target attention uses `elementwise.split_rms_norm` (dispatcher:
`torch.ops.dsv41.split_rms_norm`) for packed WQKV-A. In one launch, paired
vector workers normalize the two row segments using their own weights and
epsilons. Outputs are contiguous BF16 matrices; WQ-B, the indexer and the
KV path consume them unchanged. Both target prepare entry points share this
path. The adapter keeps the vendor norms for separate projections, older
publications, unsupported layouts/widths, and above the measured 2,048-row
range. An eager warmup prepares the JIT; for direct graph capture, use
`precompile_elementwise("split_rms_norm", rows, q_width, vector_cores,
second_width=kv_width)` before capture.

Wheel/sdist build hooks copy DeepJIT into package data, including its license
and source identity. Editable checkouts use the single root `3rdparty` tree.
Native source and CMake files are package data; host binaries are built for the
deployment environment and are not committed.

Validation commands and current coverage are documented in
[`test/ascend/dsv41/README.md`](../../../../../../test/ascend/dsv41/README.md).
