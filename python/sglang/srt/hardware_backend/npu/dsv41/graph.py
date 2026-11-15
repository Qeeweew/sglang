"""Refresh V4.1 DSpark's graph-resident batch inputs before each replay.

DSpark marks target batches as preplanned. The upstream NPU runner treats
every non-DFlash preplanned batch as having already populated its static
buffers, and only copies input IDs and positions. Our standalone attention
backend also needs current request IDs, lengths and cache locations. Reuse
the complete upstream load_batch path, including its padding policies and
replay metadata view, rather than maintain another partial buffer copier.
"""

from __future__ import annotations

import logging
from functools import wraps

from .policy import enabled

logger = logging.getLogger(__name__)


def adapt_capture_sizes(original_sizes):
    if not enabled():
        return original_sizes

    @wraps(original_sizes)
    def capture_sizes(
        model_runner, captured_req_width=1, *, gathered_buffer_required=None
    ):
        from .parallel import moe_sequence_parallel_enabled
        from sglang.srt.runtime_context import get_exec, get_parallel

        if (
            enabled()
            and model_runner.model_config.hf_config.model_type == "deepseek_v41"
            and model_runner.spec_algorithm.is_dspark()
            and moe_sequence_parallel_enabled(get_parallel())
            and (not get_exec().overlap.enable_two_batch_overlap)
        ):
            gathered_buffer_required = False
        return original_sizes(
            model_runner,
            captured_req_width,
            gathered_buffer_required=gathered_buffer_required,
        )

    return capture_sizes


def adapt_sync_batch(original_sync):
    if not enabled():
        return original_sync

    @wraps(original_sync)
    def sync_batch(self, model_runner):
        refresh_counts(self, model_runner)
        return original_sync(self, model_runner)

    return sync_batch


def refresh_counts(forward_batch, runner):
    from sglang.srt.runtime_context import get_parallel

    if (
        enabled()
        and getattr(runner.model_config.hf_config, "model_type", None) == "deepseek_v41"
        and (get_parallel().attn_dp_size > 1)
    ):
        from sglang.srt.hardware_backend.npu.dsv41.moe import live_dp_counts

        source = forward_batch.global_num_tokens_gpu
        if source is None:
            raise ValueError("resident attention DP requires device real-row counts")
        live_dp_counts(source.device, get_parallel().attn_dp_size).copy_(source)
