"""910B BF16 collectives with an explicitly prepared IPC communicator."""


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


def npu_peer_moe_gather(input, output, handle):
    """Gather owned token shards into compact EP8 ``[DP*rows,5120]`` order.

    The handle fixes attention TP2 (DP4, 1..128 local rows) or TP1 (DP8,
    1..64 local rows). BF16 local input is [rows,5120]; ranks are ordered
    DP0/TP0, DP0/TP1, DP1/TP0, ... . Odd TP2 padding is omitted.
    """
    handle.gather(input, output)
    return output


def npu_peer_moe_return(
    input, output, handle, *, shared=None, inverse=None, ids=None, weights=None
):
    """EP8 SUM to owners, shared add once, and optional TP2 restore.

    Input is BF16 ``[DP*rows,5120]``, or deferred FP16 routed values plus
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
