"""Explicit opt-in for the Ascend 910B DeepSeek V4.1 implementation.

Read before SGLang model modules are imported. Other devices keep their
existing implementations; importing this module has no device effects.
"""

import os


def enabled():
    return os.environ.get("SGLANG_DSV41_910B") == "1"
