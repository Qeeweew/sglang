# Target MoE: calibration-free H32 + integer group scaling

## Exported checkpoint and Ascend 910B serving

The target-only export is `DeepSeek-V4.1-Flash-1017-quarot-int4-g32`.
`text_config.dsv41_target_moe_format=quarot_h32_int4_g32_v1` selects it on the
910B backend.
MTP/DSpark retains its original weights and quantization. Attention, Engram
and the residual stream retain their original basis. Target conversion reads
weights only; evaluation prompts never participate in fitting.

### Joint shared/routed execution (native ABI 29)

With EP8 sequence-parallel dispatch, the first collective still gathers BF16
tokens. Each receiving rank quantizes the gathered token matrix **once**.
The local shared expert takes its owner slice, and route gather copies the
same INT8 codes and FP32 token scales into expert order. Router precision,
checkpoint layout, on-device weight storage and MTP are unchanged.

Shared W8A8 and routed G32 gate/up are separate GEMMs returning FP16. A single
clipped SwiGLU launch reads their two packed inputs directly (no concatenation)
and writes `[owned shared rows, live routed rows]` as FP16. A **separate** H32
launch quantizes that prefix, retaining one scale per row. Both operators read
the live routed count on device, including replay when no local expert fires.
The down GEMMs take slices of the resulting codes/scales. Routed down returns
FP16 for weighted combine; shared down writes BF16 directly at the model boundary.
Only routed output is weighted and combined; shared is added once on its owner.
For both G32 and source MXFP4, the EP8 serving transport leaves shared for
this joint stage even with unequal DP segments or peer communication disabled.
An idle DP with a nonempty global batch still computes its local routed experts.

### Checkpoint format

- **Weights on disk:** routed `.weight` is `[N,K/2]`, low signed nibble first;
  `.scale` is `[N,K/64+4]`. Each factor nibble is signed `t-9`; the row tail
  is its little-endian FP32 unit. The matrix kept on device consumes the same
  bytes after reordering into Cube tiles. The AIV widens signed nibbles,
  adds 9 to the block factors and multiplies exactly into INT8 for the Cube;
  full expert weights are never expanded persistently. The shared expert has
  INT8 `.weight` and FP32 `[N,1]` `.scale`, and uses `npu_quant_matmul`.
- **Basis:** let `H` be the *unnormalized* Sylvester matrix and `R=D H/√32`.
  Export `gamma' = gamma D / 4`, `W_router' = 4 W_router D`. These power-of-two
  folds preserve the router product; the selection bias is byte-for-byte unchanged.
  Gate and up encode `W R` with an extra `4/√32` in their channel units.
  For down, fold the input sign into the up projection's output rows *before*
  rounding: its symmetric clip is odd. Encode `W_down R` with `1/√32` in its
  units. Gate and up use the same output permutation in each 256-channel
  panel. After SwiGLU this gives eight interleaved, independent H32s; the
  fused H32 + token quantizer restores natural K order before the down GEMM.
  Both online transforms use FP32 butterflies, a full-row absmax and INT8
  rounding. The router and residual do not use the quantized activation.
- **Fitter:** the exporter first fits each block's continuous step, then fits
  the 16-level grid to these steps from four initial values. At the chosen
  row unit it enumerates all 16 multipliers per block, rounds the signed codes,
  and refits the row unit once using these actual products.

## Export and serve

Use a separate destination on the same filesystem as the source checkpoint:
unchanged shards are hard-linked. The exporter fits weights only; no
calibration dataset is needed. Use the same source, library and exporter to
resume a partial export. An already published destination cannot be overwritten.

```bash
ex=examples/ascend/deepseek_v41
mkdir -p build/g32-export
g++ -O3 -std=c++17 -mcpu=native -fopenmp -fno-trapping-math -shared -fPIC \
  "$ex/export_g32.cpp" -o build/g32-export/export.so
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 "$DSV41_PYTHON" \
  "$ex/export_quarot_g32.py" "$SOURCE_MODEL" "$G32_MODEL" build/g32-export/export.so
DSV41_MODEL_PATH="$G32_MODEL" bash "$ex/serve.sh"
```

### G32 GEMM: overlapping output correction with the next weight panel

The G32 kernel keeps the weight decode buffers live across output tiles. On
the target projections, the AIVs publish the first K panel of a later tile
before applying the previous tile's channel/token scales. After reading the
previous INT32 result into UB, they issue the next weight DMA, then compute
`float32(dot) * channel * token` and round once to the output dtype. The
remaining K panels depend on that DMA: the cube can start ahead, but cannot
overwrite the old result slot before its AIV readers are done. The output DMA
is retired at UB reuse, not before starting the next tile. There is no extra
kernel launch, result ring or change to the repacked weights.

If a complete activation tile and two B panels fit together in L1 (the down
projection), adjacent N tiles with the same input row range reuse A. For a
more-than-half-full padded tile on this path, FIX drains all MMAD row
fragments into private scratch. Input reads and final stores still use only
valid rows. Shortening the FIX transfer to the valid rows can hang a later
tile with a different shape, even if the first tile looks correct.

For synthetic routed GEMM latency, run `test/manual/kernels/bench_npu_g32_routing.py`
with `--output` under `build/perf/`. It does not load a checkpoint.
