"""FP32 DSpark scheduling through the native Torch binding."""

from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    NativeOperator,
    load_library,
)


class DSparkSchedule(NativeOperator):
    def __init__(self, *, library=None, device="npu:0"):
        super().__init__(device)
        self.library = load_library(native_library_path(library))

    @staticmethod
    def workspace_shape(rows, gamma, max_verify_len=0):
        return min(max_verify_len or gamma + 1, gamma), (rows + 7) // 8 * 8

    def bind(
        self,
        confidence,
        budget,
        *,
        min_verify_len=1,
        max_verify_len=0,
        survival_eps=1e-6,
        survival=None,
        out,
    ):
        """Prepare the actual physical layout before capture; retain fixed buffers."""
        from functools import partial

        args = (
            confidence,
            budget,
            min_verify_len,
            max_verify_len,
            survival_eps,
            survival,
            out,
        )
        self.library.prepare_dspark_schedule(*args)
        return partial(self.library.dspark_schedule, *args)

    def __call__(
        self,
        confidence,
        budget,
        *,
        min_verify_len=1,
        max_verify_len=0,
        survival_eps=1e-6,
        survival=None,
        out=None,
    ):
        return self.library.dspark_schedule(
            confidence,
            budget,
            min_verify_len,
            max_verify_len,
            survival_eps,
            survival,
            out,
        )
