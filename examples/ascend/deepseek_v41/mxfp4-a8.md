# Routed MXFP4 × per-token INT8 on 910B

On the opt-in V4.1 910B backend, set `SGLANG_OPT_DSV41_NPU_TARGET_MOE_A8=1`
before startup to use this GEMM for target routed experts (prefill and decode).
It dynamically quantizes the gathered FP16 input rows for gate/up and the
clipped SwiGLU output rows for down separately. The default is off. Routing,
shared/dense weights and MTP/draft keep their existing paths.

## Packing contract

`pack_expert(codes, scales)` takes one expert on CPU: raw checkpoint bytes
`uint8[N,K/2]` (E2M1, low nibble first) and `uint8[N,K/32]` (E8M0). Supported
K is a multiple of 32 in [256, 8192]; N is a positive multiple of 64 below
65536. Both gate/up (concatenate on N) and down use the same operator.

For a row with raw group exponents `E`, choose `b = max(min(E), max(E)-3)`
and store one FP32 `s_w = 2**(b-128)`. For groups in range, the INT8 Cube
operand is `q_w = (2 * E2M1) * 2**(E-b)`. The first factor has absolute
values 0, 1, 2, 3, 4, 6, 8, 12; the second is 1, 2, 4 or 8. Thus `|q_w| <= 96`
and `q_w * s_w` recovers the checkpoint value exactly. The factor of two
preserves half-integer FP4 levels. For example, group scales `1/8, 1/4, 1/2, 1`
give `s_w=1/16`; an FP4 value of `-1.5` in the `1/2` group becomes INT8 `-12`.

For a wider row the base is `max(E)-3`. If `E < b`, store group factor 1
and truncate the **absolute** doubled FP4 level by shifting it right by `b-E`
before restoring the sign. After a shift the result is still in the same
eight-level alphabet, so it still takes four bits including sign. The absolute
error is less than one `s_w`, and the magnitude never increases (including for
negative values). All other groups are exact. `allow_rounding=False` rejects
any weight that would change. NaN exponents and subnormal channel coefficients
are unsupported in either mode. The CPU packer uses intact nibbles directly
and only looks up replacements for under-range groups. Either sign of FP4
zero decodes to INT8 zero.

The packed output contains four bits per weight, four bits per K32 group and
four bytes per channel. The group factor is stored as signed **negative** INT4
(`-1/-2/-4/-8`), so the decoder can use a single signed vector conversion.
Per expert this is `NK/2 + NK/64 + 4N` bytes, no padding, versus checkpoint
`NK/2 + NK/32`. At K ≥ 256 the resident representation does not grow.

## Execution

AIV decodes weight panels into a double-buffered INT8 workspace; Cube
accumulates the full K dot product in INT32. AIV multiplies by the FP32
channel and token scales before converting to FP16. Only packed weight codes
bypass L2; group factors, scales and the inter-core workspace retain their
normal cache policy. There is no full-size expanded weight tensor.

Use `test/manual/kernels/bench_npu_mxfp4_a8.py` for local comparisons. Store
measurements in `build/perf/`, outside version control.
