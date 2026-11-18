"""V4.1 DSpark proposal, scheduling and accepted-prefix commits on 910B.

Keep SGLang's worker, proposal, acceptance, bonus and request accounting. The
target verifier publishes accepted lengths before the existing Engram history
commit; the same lengths commit the target backend's provisional C1/C2 state.
"""

from __future__ import annotations

import logging
from functools import cache, wraps

import torch

from ..policy import enabled

logger = logging.getLogger(__name__)


def _attention_tp_enabled(worker):
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    return (
        torch.device(worker.device).type == "npu"
        and worker.model_runner.model_config.hf_text_config.model_type == "deepseek_v41"
        and p.tp_size == 8
        and p.attn_dp_size == 4
        and p.attn_tp_size == 2
        and p.attn_cp_size == 1
        and p.moe_ep_size == 8
    )


@cache
def _schedule(device, stream, rows, gamma, maximum):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import DSparkSchedule

    operator = DSparkSchedule(device=device)
    output = torch.empty(rows, device=device, dtype=torch.int32)
    return operator, output


def _target_state_commit(backend, lengths, *, folded=False):
    commit = getattr(backend, "commit_after_verify", None)
    if commit is not None:
        if folded:
            commit(lengths, folded=True)
        else:
            commit(lengths.to(torch.int32).contiguous())
        return
    raise RuntimeError("910B V4.1 target backend is missing commit_after_verify")


def _install_verify_epilogue(worker):
    """Use SGLang's static epilogue and its ordinary eligibility/fallback rules."""
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        CommitInjectCtx,
        DsparkVerifyEpilogue,
    )

    if (
        not worker._hosts_draft
        or torch.device(worker.device).type != "npu"
        or not worker._verify_executor._target_is_dsv41
        or worker._verify_epilogue is not None
        or worker._verify_planner.mode_value != "static"
        or not worker._decode_graph_allowed
        or not worker._draft_is_moe
        or get_parallel().attn_dp_enabled
        or get_parallel().pp_size != 1
    ):
        return

    class VerifyEpilogue(DsparkVerifyEpilogue):
        commits_target_state = True

        def _accept(self, **kwargs):
            from sglang.srt.hardware_backend.npu.dsv41.dspark.verify import (
                _commit_sources,
            )

            commit_lens = super()._accept(**kwargs)
            meta = worker.model_runner.attn_backend.forward_metadata
            # The upstream gate disables graph side effects for eager sampling
            # and grammar steps. Dummy requests must not commit any payload.
            commit_lens.mul_((meta.verify_requests != 0).int())
            meta.verify_commit_lens.copy_(commit_lens * self.inject_gate_buf)
            _commit_sources(meta)
            return commit_lens

    epilogue = VerifyEpilogue(
        max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
        verify_num_draft_tokens=worker.verify_num_draft_tokens,
        device=worker.device,
        tp_sync=worker._tp_sync,
        fused_argmax=True,
        simulated_correct_len=worker._simulated_correct_len,
        commit_ctx=CommitInjectCtx(
            draft_model=worker.draft_model,
            block_pos_offsets=worker._block_pos_offsets,
            resolve_pool=lambda: worker.draft_model_runner.token_to_kv_pool,
            resolve_req_to_token=lambda: (
                worker.model_runner.req_to_token_pool.req_to_token
            ),
            kv_injector=worker._kv_injector,
        ),
    )
    worker._verify_epilogue = worker._verify_executor.verify_epilogue = epilogue
    worker.model_runner.capture_tail_hooks.append(epilogue.capture_hook)
    logger.info(
        "V4.1 DSpark: capture accept, C1/C2 commit and draft KV injection in target Graph"
    )


def adapt_markov(original_markov):
    if not enabled():
        return original_markov

    @wraps(original_markov)
    def markov(cls, **kwargs):
        if kwargs["base_logits"].device.type == "npu":
            from .markov import AscendMarkovStep

            return AscendMarkovStep.run(**kwargs)
        return original_markov(cls, **kwargs)

    return markov


def adapt_greedy(original_greedy):
    if not enabled():
        return original_greedy

    @wraps(original_greedy)
    def greedy(cls, **kwargs):
        if kwargs["base_logits"].device.type == "npu":
            return cls.torch(**kwargs)
        return original_greedy(cls, **kwargs)

    return greedy


def adapt_schedule(original_schedule):
    if not enabled():
        return original_schedule

    @wraps(original_schedule)
    def schedule(cls, *, confidence, budget, cfg):
        if confidence.device.type != "npu":
            return original_schedule(cls, confidence=confidence, budget=budget, cfg=cfg)
        rows, gamma = confidence.shape
        maximum = cfg.resolved_max_verify_len()
        op, output = _schedule(
            str(confidence.device),
            int(torch.npu.current_stream(confidence.device).npu_stream),
            rows,
            gamma,
            maximum,
        )
        return op(
            confidence.float().contiguous(),
            int(budget),
            min_verify_len=cfg.min_verify_len,
            max_verify_len=maximum,
            survival_eps=cfg.survival_eps,
            out=output,
        )

    return schedule


def adapt_accept(original_accept):
    if not enabled():
        return original_accept

    @wraps(original_accept)
    def accept(self, *args, **kwargs):
        result = original_accept(self, *args, **kwargs)
        if self._target_is_dsv41 and result.commit_lens.device.type == "npu":
            _target_state_commit(
                self.model_runner.attn_backend,
                result.commit_lens,
                folded=kwargs.get("folded_accept", False)
                and getattr(self.verify_epilogue, "commits_target_state", False),
            )
        return result

    return accept


def adapt_worker_init(original):
    if not enabled():
        return original

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _install_verify_epilogue(self)

    return initialize
