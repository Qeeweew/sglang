"""910B implementation of the local candidate / gather / winner contract."""

from functools import cache

import torch
import torch_npu

from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

from ..kernels.runtime.native import load_library


@cache
def _operators():
    ops = load_library()
    return getattr(ops, "markov_local_greedy", None), getattr(
        ops, "markov_merge_candidates", None
    )


class AscendMarkovStep(MarkovSampleStep):
    @classmethod
    def run(cls, **args):
        base, bias = args["base_logits"], args["bias"]
        local, _ = _operators()
        if (
            local is None
            or args.get("has_sampled_rows") is not False
            or args.get("logits_out") is not None
            or args["step_dtype"] != torch.float32
            or any(
                tensor.dtype not in (torch.bfloat16, torch.float32)
                or torch_npu.get_npu_format(tensor) not in (0, 2)
                for tensor in (base, bias)
            )
        ):
            return cls.torch(**args)
        gather = args.get("gather")
        start = args.get("vocab_start", 0)
        cls._check(base, bias, args["temperatures"], args["offset"], args["step_dtype"],
                   start, gather, args.get("gather_logits"), None)
        packet = local(base, bias, start)
        return cls.finish_candidates(packet if gather is None else gather(packet), base.shape[0])

    @staticmethod
    def finish_candidates(gathered, rows):
        if not rows:
            return gathered.new_empty((0,), dtype=torch.int64)
        _, merge = _operators()
        if merge is not None and gathered.is_contiguous() and torch_npu.get_npu_format(gathered) in (0, 2):
            return merge(gathered.view(-1, rows, 2))
        return MarkovSampleStep.finish_candidates(gathered, rows)
