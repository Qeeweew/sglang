"""910B collectives with an explicitly prepared IPC communicator."""


def npu_peer_all_reduce(input, output, handle):
    """Sum BF16 ``[1..128, 5120]`` across two same-host 910B ranks.

    ``handle`` is a prepared native PeerCommunicator resource owned by the caller.
    Input and output must be contiguous ND tensors with distinct storage. Calls
    sharing a handle must be serialized, including Graph replay on other streams.
    The caller retains the handle until all captured graphs have been retired.
    FP32 addition followed by round-to-nearest BF16 preserves IEEE Inf/NaN.
    """
    handle.reduce(input, output)
    return output


def npu_peer_group_gather(
    input, output, handle, *, other=None, other_output=None, skip_own_output=False
):
    """Gather two same-host 910B ranks into contiguous scheduler row order.

    Tensors are ND [local_rows, D] or [outer, local_rows, D], with 32-byte
    aligned addresses and row widths. Outputs replace local_rows with valid_rows.
    Rank 0 provides local_rows and rank 1 provides valid_rows - local_rows.
    The optional second input/output pair can have a different width and dtype.
    With skip_own_output, only the peer's rows of each output are written.
    Warm one launch eagerly before capture and serialize uses of the handle.
    """
    handle.group_gather(
        input, other, output, other_output, output.shape[-2], skip_own_output
    )
    return output, other_output


def npu_peer_moe_return(
    input, output, handle, *, shared=None, inverse=None, ids=None, weights=None
):
    """EP8 SUM to owners, shared add once, and optional TP2 restore.

    Input is BF16 ``[DP*rows,5120]``, or deferred BF16/FP16 routed values plus
    int32 inverse/IDs and FP32 route weights. Output is BF16 ``[rows,5120]``.
    DP8 has no TP restore. Shared input covers the owned local shard.
    Rank-ordered FP32 SUM is BF16-rounded before the optional shared addition.
    Serialized calls, handle lifetime and device requirements match all-reduce.
    """
    if inverse is None:
        handle.return_sum(input, shared, output)
    else:
        handle.finalize_return(input, inverse, ids, weights, shared, output)
    return output


def npu_peer_post(
    input,
    residual,
    pre,
    post,
    comb,
    weight,
    updated,
    normalized,
    handle,
    *,
    inverse=None,
    ids=None,
    weights=None,
    shared=None,
):
    """TP2 SUM or EP8 route finalize/return, fused with four-branch mHC post.

    TP2 includes collapse and RMSNorm; EP8 also supports post-only terminal
    boundaries. Both reuse the native mHC arithmetic pipeline, including its
    BF16 intermediate boundaries and FP32 coefficient order.
    All tensors are on one 910B; residual/updated are BF16 [rows,4,5120],
    coefficients and optional norm weight are FP32. Resources are preprepared.
    """
    handle.post(
        input,
        residual,
        pre,
        post,
        comb,
        weight,
        updated,
        normalized,
        inverse,
        ids,
        weights,
        shared,
    )
    return updated, normalized
