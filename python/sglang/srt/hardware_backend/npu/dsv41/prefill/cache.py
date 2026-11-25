"""910B V4.1 BF16 KV and INT8 Indexer pools using common layer ownership.

The model's C1/C2 storage remains addressed by ``full_loc // ratio``. It does
not allocate the legacy NPU C128 sidecar allocator or CUDA FP4 index buffers.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from functools import wraps

from ..policy import enabled

logger = logging.getLogger(__name__)
_pool_is_draft = ContextVar("dsv41_pool_is_draft", default=False)


def adapt_validator(original_validator):
    if not enabled():
        return original_validator

    @wraps(original_validator)
    def validator(component, match_device_only=False):
        pool = component.cache.token_to_kv_pool_allocator.get_kvcache()
        if getattr(pool, "_dsv41_910b", False) and pool.request_window is not None:
            return lambda node: not node.evicted
        return original_validator(component, match_device_only)

    return validator


def adapt_radix_init(original):
    if not enabled():
        return original

    @wraps(original)
    def init(self, params):
        pool = params.token_to_kv_pool_allocator.get_kvcache()
        if getattr(pool, "_dsv41_910b", False) and 128 not in pool.compression_ratios:
            # The cache factory derives typed tree slots from the native pools.
            # They own C1/C2 through FULL pages and have no C128 sidecar. Keep
            # those slots and family overrides intact; tree_components is now
            # a read-only view of tree_slots.
            if params.page_size != 128:
                raise ValueError("910B V4.1 prefix caching requires 128-token pages")
        original(self, params)

    return init


def adapt_match(original_match):
    if not enabled():
        return original_match

    @wraps(original_match)
    def match(self, params):
        from dataclasses import replace

        result = original_match(self, params)
        pool = self.token_to_kv_pool_allocator.get_kvcache()
        if getattr(pool, "_dsv41_910b", False) and pool.request_window is not None:
            return result._replace(swa_branching_seqlen=None)
        coverage = getattr(pool, "_dsv41_ced_coverage", None)
        if coverage is None:
            return result
        initial = result.device_prefix_len
        while result.device_prefix_len:
            full = self.path_device_indices(result.last_device_node)[
                : result.device_prefix_len
            ]
            swa = pool.translate_loc_from_full_to_swa(full.long())
            bound = coverage.reusable_prefix(full, swa, page=self.page_size)
            if bound == result.device_prefix_len:
                break
            result = original_match(self, replace(params, key=params.key[:bound]))
        self.ced_rejected_prefix_tokens = (
            getattr(self, "ced_rejected_prefix_tokens", 0)
            + initial
            - result.device_prefix_len
        )
        return result._replace(swa_branching_seqlen=None)

    return match


def adapt_checkpoint(original_checkpoint):
    if not enabled():
        return original_checkpoint

    @wraps(original_checkpoint)
    def checkpoint(self, req, *, up_to, **kwargs):
        pool = self.token_to_kv_pool_allocator.get_kvcache()
        coverage = getattr(pool, "_dsv41_ced_coverage", None)
        if coverage is not None:
            req.swa_branching_seqlen = None
            if not req.finished():
                self.advance_unpublished_req(req)
                return
        elif (
            getattr(pool, "_dsv41_910b", False)
            and pool.request_window is not None
            and pool.needs_paged_swa_allocator
            and self.decoder_swa_bounded_replay
            and not self.disable
            and not req.finished()
        ):
            from .replay import CED_KEEP_TOKENS

            # A canonical FULL prefix can predate draft KV injection. Keep
            # the pages the running draft will read on its private locations.
            prompt_end = max(req.extend_end, len(req.origin_input_ids))
            up_to = min(up_to, max(0, prompt_end - CED_KEEP_TOKENS))
            if up_to < req.kv.cache_protected_len + self.page_size:
                req.swa_branching_seqlen = None
                self.advance_unpublished_req(req)
                return
        return original_checkpoint(self, req, up_to=up_to, **kwargs)

    return checkpoint


def adapt_fast(original_fast):
    if not enabled():
        return original_fast

    @wraps(original_fast)
    def fast(self):
        pool = self.token_to_kv_pool_allocator.get_kvcache()
        if getattr(pool, "_dsv41_ced_coverage", None) is not None or (
            getattr(pool, "_dsv41_910b", False) and pool.request_window is not None
        ):
            return False
        return original_fast(self)

    return fast


def adapt_build(original_build):
    if not enabled():
        return original_build

    @wraps(original_build)
    def build(self, **kwargs):
        token = _pool_is_draft.set(self.is_draft_worker)
        try:
            return original_build(self, **kwargs)
        finally:
            _pool_is_draft.reset(token)

    return build


def adapt_pool_plan(original):
    if not enabled():
        return original

    @wraps(original)
    def init(self, kvc):
        original(self, kvc)
        config = kvc.model_config.hf_config
        if getattr(config, "model_type", None) != "deepseek_v41":
            return
        previous = self.kv_bytes
        self.kv_bytes = self.attn_head_dim * 2
        self.indexer_bytes_per_token = self.indexer_head_dim + 2
        self.low_ratio_index_bytes = self.indexer_head_dim + 2
        if self.encoder_replay:
            slots = self.requested_max_running_requests_per_worker + 1
            capacity = (
                (
                    self.sliding_window_size
                    + self.online_c128_mtp_max_draft_tokens
                    + self.page_size
                    - 1
                )
                // self.page_size
                * self.page_size
            )
            from sglang.srt.runtime_context import get_schedule

            scratch = (
                max(
                    get_schedule().chunked_prefill_size,
                    slots * max(128, self.online_c128_mtp_max_draft_tokens),
                )
                + slots * 128
                + self.page_size
            )
            self.request_window_bytes += (
                (slots * capacity + self.page_size) * self.num_layers_total
                + 4 * scratch
            ) * (self.kv_bytes - previous)
        self.bytes_per_swa_token = self._get_bytes_per_swa_token() * self._spec_infl
        self.bytes_per_full_token = self._get_bytes_per_full_token()
        if self._spec_infl != 1:
            compressed = sum(
                (
                    self._compressed_bytes_per_full_token(ratio) * len(layers)
                    for ratio, layers in self.stage_owner_layers.items()
                )
            )
            self.bytes_per_full_token += compressed * (self._spec_infl - 1)

    return init
