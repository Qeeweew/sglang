"""Model geometry used by offline specialization and SGLang integration.

Kernel bodies receive dimensions as runtime or constexpr arguments. Buffer tile
sizes and storage-format constants belong to the individual kernel algorithms.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelGeometry:
    hidden: int = 5120
    intermediate: int = 2304
    experts: int = 384
    mhc_streams: int = 4
    head_dim: int = 512
    attention_heads: int = 64
    wo_groups: int = 8
    wo_k: int = 4096
    wo_rank: int = 1024

    @property
    def mhc_width(self):
        return self.hidden * self.mhc_streams


V41 = ModelGeometry()


def specialization(name):
    """Default model shapes; explicit builder parameters can override them."""
    if name.startswith("mhc_"):
        return {"D": V41.hidden}
    return {
        "rms_norm": {"D": V41.hidden},
        "clipped_swiglu": {"D": V41.intermediate},
        "dense_projection": {"K": V41.hidden, "COLUMNS": V41.head_dim},
        "wo_a_projection": {
            "K": V41.wo_k,
            "RANK": V41.wo_rank,
            "GROUP_COUNT": V41.wo_groups,
        },
    }.get(name, {})
