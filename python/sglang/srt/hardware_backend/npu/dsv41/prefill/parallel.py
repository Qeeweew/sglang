"""Host-known row geometry for eager V4.1 bounded replay on resident ranks."""

import torch


def gather_dp_rows(local_rows):
    """Gather on TP's existing CPU group, then collapse attention-TP replicas.

    All target ranks participate, including idle and verify peers. The CPU
    collective neither synchronizes NPU work nor creates a new process group.
    """
    from sglang.srt.distributed.utils import all_gather_single
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    values = torch.empty(p.tp_size, dtype=torch.int64)
    all_gather_single(
        values,
        torch.tensor([local_rows], dtype=torch.int64),
        group=p.tp_group.cpu_group,
    )
    replicas = values.view(p.attn_dp_size, p.attn_tp_size)
    if not bool((replicas == replicas[:, :1]).all()):
        raise RuntimeError("V4.1 bounded replay row counts differ within attention TP")
    return replicas[:, 0].tolist()


def late_layer_dp_counts(fb, local_rows, *, in_graph=False):
    if in_graph:
        return None
    early = fb.global_num_tokens_padded_cpu
    if early is None:
        early = fb.global_num_tokens_cpu
    if early is None or torch.npu.is_current_stream_capturing():
        return None
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    mode = getattr(fb, "forward_mode", None)
    can_skip = (
        (p.attn_dp_size, p.attn_tp_size) in ((4, 2), (8, 1))
        and mode is not None
        and (
            mode.is_idle()
            or (
                mode.is_extend_without_speculative()
                and getattr(fb, "_dsv41_ced_empty_chunk", False)
            )
        )
    )
    # Share the decision in the existing collective. A verify/decode peer
    # must keep every EP rank in the consumer-layer collective sequence.
    packed = 2 * local_rows + int(not can_skip)
    values = [packed] if len(early) == 1 else gather_dp_rows(packed)
    counts = [value // 2 for value in values]
    fb._dsv41_ced_skip_all = all(value % 2 == 0 for value in values)
    if counts != list(early) and not fb._dsv41_ced_skip_all:
        # Upload before layer 0, rather than blocking queued source-layer work.
        local = counts[p.attn_dp_rank if len(counts) > 1 else 0]
        real = min(local, fb.global_num_token_non_padded_cpu)
        fb._dsv41_ced_dp_inputs = (
            tuple(counts),
            torch.tensor(
                counts, dtype=torch.int64, device=fb.global_num_tokens_gpu.device
            ),
            real,
            None
            if fb.global_num_token_non_padded is None
            else fb.global_num_token_non_padded.new_tensor(real),
        )
    return None if counts == list(early) else counts


def enter_staged_dp(fb, counts):
    """Switch host geometry and device views without a transfer at the cut."""
    from sglang.srt.layers.dp_attention import (
        DpPaddingMode,
        set_dp_buffer_len_from_batch,
    )
    from sglang.srt.models.deepseek_common.decoder_swa_bounded_replay import (
        enter_late_layer_dp,
    )

    staged = getattr(fb, "_dsv41_ced_dp_inputs", None)
    if staged is None or staged[0] != tuple(counts):
        return enter_late_layer_dp(fb, counts)
    # Keep the upstream exit_late_layer_dp's 12-field restoration contract.
    names = (
        "global_num_tokens_cpu",
        "global_num_tokens_gpu",
        "global_num_tokens_padded_cpu",
        "global_num_tokens_for_logprob_cpu",
        "global_num_tokens_for_logprob_gpu",
        "global_dp_buffer_len",
        "dp_padding_mode",
        "dp_local_start_pos",
        "dp_local_num_tokens",
        "global_num_token_non_padded",
        "global_num_token_non_padded_cpu",
        "num_token_non_padded",
    )
    saved = tuple(getattr(fb, name) for name in names)
    fb.global_num_tokens_cpu = fb.global_num_tokens_padded_cpu = counts
    fb.global_num_tokens_gpu = staged[1]
    fb.global_num_tokens_for_logprob_cpu = counts
    fb.global_num_tokens_for_logprob_gpu = staged[1]
    fb.global_dp_buffer_len = sum(counts)
    fb.dp_padding_mode = DpPaddingMode.SUM_LEN
    fb.dp_local_start_pos = fb.dp_local_num_tokens = None
    fb.global_num_token_non_padded_cpu = staged[2]
    if staged[3] is not None:
        fb.global_num_token_non_padded = fb.num_token_non_padded = staged[3]
    set_dp_buffer_len_from_batch(fb)
    return saved


def prepare_balanced_tail(fb, counts):
    """Retain the qualified MoE-SP path when shrinking needs no padding."""
    from sglang.srt.layers.dp_attention import (
        DpPaddingMode,
        set_dp_buffer_len_from_batch,
    )
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    if (
        (p.attn_dp_size, p.attn_tp_size) in ((4, 2), (8, 1))
        and p.moe_ep_size == 8
        and len(counts) == p.attn_dp_size
        and counts[0] > 0
        and counts[0] % p.attn_tp_size == 0
        and len(set(counts)) == 1
    ):
        fb.dp_padding_mode = DpPaddingMode.MAX_LEN
        set_dp_buffer_len_from_batch(fb)


def prepare_replay_dp(fb, runner):
    """Publish merged query counts before SGLang chooses TP/DP padding.

    Operate on ForwardBatch: scheduler token ownership and DSpark coordination
    stay unchanged, and verify peers have already applied speculative scaling.
    Logits counts describe requested outputs, so they remain unchanged.
    """
    from sglang.srt.runtime_context import get_parallel

    if (
        runner.token_to_kv_pool.request_window is not None
        and get_parallel().attn_dp_size > 1
        and fb.is_extend_in_batch
        and fb.global_num_tokens_cpu is not None
        and len(fb.global_num_tokens_cpu) > 1
    ):
        counts = gather_dp_rows(fb.input_ids.numel())
        fb.global_num_tokens_cpu = counts
        fb.global_num_tokens_gpu.copy_(
            torch.tensor(counts, device=fb.global_num_tokens_gpu.device)
        )


def enter_dp(fb, counts):
    saved = enter_staged_dp(fb, counts)
    fb._dsv41_ced_saved_dp = saved
    prepare_balanced_tail(fb, counts)
    return saved


def exit_dp(fb, saved):
    from sglang.srt.models.deepseek_common.decoder_swa_bounded_replay import (
        exit_late_layer_dp,
    )

    exit_late_layer_dp(fb, saved)
    fb._dsv41_ced_saved_dp = None
