"""Device-selected DSpark page metadata; preserve full-context table storage."""

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    NativeOperator,
    load_library,
)


class DraftPageMetadata(NativeOperator):
    """Write only pages read by draft SWA; other table entries remain untouched.

    The caller owns a persistent full-context page table. Consumers must use the
    returned lengths and per-request query boundaries: only the final
    window_rows prefix tokens plus the complete draft block are translated. This does not
    reduce the model context or change the token allocator's physical mapping.
    """

    def __init__(self, *, library=None, device="npu:0"):
        super().__init__(device)
        self.library = load_library(native_library_path(library))
        self._prepared = set()
        self._worker_budget = torch.npu.get_device_properties(
            self.device
        ).vector_core_num

    def __call__(
        self,
        req_to_token,
        full_to_swa,
        requests,
        prefix,
        full_slots,
        *,
        cu,
        lengths,
        table,
        slots,
        routed_rows,
        block,
        page_rows=128,
        window_rows=128,
    ):
        args = (
            req_to_token,
            full_to_swa,
            requests,
            prefix,
            full_slots,
            cu,
            lengths,
            table,
            slots,
            routed_rows,
            block,
            page_rows,
            window_rows,
        )
        workers = min(self._worker_budget, max(1, (requests.numel() + 7) // 8))
        key = (
            *req_to_token.shape,
            full_to_swa.numel(),
            page_rows,
            window_rows,
            workers,
            requests.numel(),
            block if requests.numel() else 0,
        )
        if key not in self._prepared:
            self.library.prepare_draft_page_metadata(*args)
            self._prepared.add(key)
        self.library.draft_page_metadata(*args)
