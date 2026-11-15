"""BF16 attention and INT8 Indexer pool implementations for 910B."""

import weakref

import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.dsv4 import dsv4_memory_pool as module
from sglang.srt.hardware_backend.npu.dsv4 import dsv4_req_to_token_pool as req_module
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

from sglang.srt.hardware_backend.npu.dsv41.prefill.cache import _pool_is_draft


class BF16KVPool(module.NPUDeepSeekV4SingleKVPool):
    def get_bytes_per_token(self):
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) * 2

    def create_buffer(self, *, num_pages):
        if self.dtype != torch.bfloat16:
            raise ValueError("910B V4.1 attention KV requires BF16")
        dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_cache_total_dim = dim
        self.bytes_per_page_padded = self.kernel_page_size * dim * 2
        # The FULL allocator reserves logical page 0. Compressed addresses
        # reserve the correspondingly divided local page as well.
        pages = (
            self.size + self.page_size + self.kernel_page_size - 1
        ) // self.kernel_page_size
        return torch.zeros(
            (pages, self.kernel_page_size, 1, dim),
            dtype=torch.bfloat16,
            device=self.device,
        )


class INT8IndexerPool(module.NPUDeepSeekV4IndexerPool):
    def _create_buffer(self):
        # Deliberately bypass both the common packed-CUDA allocation and
        # the old NPU implementation that allocated a redundant copy.
        kp = self._kernel_page_size
        pages = (self.size + self.page_size + kp - 1) // kp
        self.index_k_with_scale_buffer = None
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.index_k_buffer = [
                torch.zeros(
                    (pages, kp, 1, self.index_head_dim),
                    dtype=torch.int8,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.index_scale_buffer = [
                torch.zeros((pages, kp, 1, 1), dtype=torch.float16, device=self.device)
                for _ in range(self.layer_num)
            ]

    def get_bytes_per_token(self):
        return self.index_head_dim + 2

    def contiguous_page_row_buffers(self):
        return [
            x.view(torch.uint8).flatten(1)
            for x in (*self.index_k_buffer, *self.index_scale_buffer)
        ]

    def get_index_k_with_scale_buffer(self, layer_id):
        raise RuntimeError("910B Indexer uses separate INT8 K and FP16 scale buffers")


class DSV41NPUTokenToKVPool(module.DSV4NPUTokenToKVPool):
    _dsv41_910b = True

    def __init__(self, *args, **kwargs):
        # The upstream NPU configurator omits this CUDA-base constructor
        # argument. Preserve the real builder role, especially under replay.
        kwargs.setdefault("is_draft_worker", _pool_is_draft.get())
        ratios = kwargs.get("compression_ratios")
        if ratios is None or not set(ratios).issubset({0, 1, 2}):
            raise ValueError("910B V4.1 cache integration serves ratios 0/1/2")
        if kwargs.get("dtype") != torch.bfloat16:
            raise ValueError("910B V4.1 KV cache must be BF16")
        # Retain common V4.1 source ownership, layer mapping, pair-state
        # sizing and request-window support; skip legacy C128 validation.
        self._dsv41_state_owners = weakref.WeakSet()
        self._dsv41_ced_coverage = None
        self.c128_page_size = 64
        DeepSeekV4TokenToKVPool.__init__(self, *args, **kwargs)
        from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import ced_enabled

        if (
            ced_enabled()
            and self.request_window is None
            and not kwargs["is_draft_worker"]
        ):
            from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import (
                WindowCoverage,
            )

            self._dsv41_ced_coverage = WindowCoverage(self.full_size + self.page_size)
        self.mem_usage = self.get_kv_size_bytes() / 2**30

    def register_request_state_owner(self, owner):
        self._dsv41_state_owners.add(owner)

    def clear_request_scoped_state(self, req_pool_idx):
        super().clear_request_scoped_state(req_pool_idx)
        # Native projected compressors use stable buffers owned by the
        # backend, separate from the upstream pair-state pool. Release both
        # together before this request slot can be handed to another prompt.
        for owner in self._dsv41_state_owners:
            owner.clear_request_state(req_pool_idx)

    def get_kv_size_bytes(self):
        buffers = []
        if self.swa_kv_pool is not None:
            buffers.extend(self.swa_kv_pool.kv_buffer)
        for pool in self.kv_pools.values():
            if pool is not None:
                buffers.extend(pool.kv_buffer)
        for pool in self.index_pools.values():
            buffers.extend(pool.index_k_buffer)
            buffers.extend(pool.index_scale_buffer)
        for pool in self.compress_state_pools + self.indexer_compress_state_pools:
            if pool is not None:
                buffers.append(pool.kv_score_buffer.kv_score)
        if self.request_window is not None:
            buffers.extend(self.request_window.state.kv_buffer)
            if self.request_window.workspace is not None:
                buffers.extend(self.request_window.workspace.kv_buffer)
        # Source-layer sharing is already represented by one pool buffer;
        # deduplicate any remaining tensor views by their storage address.
        unique = {
            x.untyped_storage().data_ptr(): x.untyped_storage().nbytes()
            for x in buffers
        }
        return sum(unique.values())

    def get_swa_buffer(self, layer_id, loc=None):
        if self.request_window is None:
            return super().get_swa_buffer(layer_id, loc)
        value = self.request_window.buffer(self._swa_local_layer_id(layer_id))
        return value if loc is None else value.flatten(0, 1)[loc]

    def get_swa_key_bytes_per_token(self):
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) * 2

    def _make_kv_pool(
        self,
        *,
        size,
        page_size,
        dtype,
        layer_num,
        device,
        enable_memory_saver,
        global_page_size,
        **kwargs,
    ):
        return BF16KVPool(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            global_page_size=global_page_size,
            kernel_page_size=page_size,
        )

    def _make_indexer_pool(
        self,
        size,
        page_size,
        dtype,
        index_head_dim,
        layer_num,
        device,
        enable_memory_saver,
        force_fp4=False,
        global_page_size=None,
        index_k_page8=False,
    ):
        if index_k_page8:
            raise ValueError("910B Indexer requires 16-aligned PA_ND pages")
        if force_fp4:
            ratio = (self.full_size + self.page_size) // size
            if ratio not in (1, 2):
                raise ValueError(f"unexpected V4.1 indexer allocation ratio {ratio}")
            # Align index and main-cache physical pages for FULL-page
            # ownership: C1=128 rows, C2=64 rows at page_size=128.
            page_size = self.page_size // ratio
        if page_size < 16 or page_size % 16:
            raise ValueError("INT8 QLI requires a page size divisible by 16")
        return INT8IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=page_size,
            global_page_size=page_size,
            use_fp4_indexer=False,
        )

    def get_key_buffer(self, layer_id):
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return self.get_swa_buffer(layer_id)
        return item.compress_kv_pool.kv_buffer[item.compress_layer_id]

    def get_contiguous_buf_infos(self):
        buffers = []
        for ratio, pool in self.kv_pools.items():
            if pool is not None:
                buffers.extend(pool.kv_buffer)
            index = self.index_pools.get(ratio)
            if index is not None:
                buffers.extend(index.index_k_buffer)
                buffers.extend(index.index_scale_buffer)
        return (
            [x.data_ptr() for x in buffers],
            [x.nbytes for x in buffers],
            [x[0].nbytes for x in buffers],
        )

    def set_compress_buffer(self, layer_id, loc, kv, kv_scale, from_indexer):
        item = self.layer_mapping[layer_id]
        if from_indexer:
            self.index_pools[item.compress_ratio].set_index_k_scale(
                item.compress_layer_id, loc, kv, kv_scale
            )
            return
        target = item.compress_kv_pool.kv_buffer[item.compress_layer_id].flatten(0, 1)
        target[loc.long()] = kv.to(target.dtype).reshape(-1, 1, target.shape[-1])


class DSV41NPUTokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pool = self.get_kvcache()
        if pool.request_window is not None and not pool.needs_paged_swa_allocator:
            # Without DSpark no paged SWA storage is needed. The NPU
            # configurator does not pass req_to_token_pool here; its
            # registration below completes the existing ring accounting.
            self._swa_req_ring = True
            self._swa_ring_cost = pool.request_window.capacity

    def free(self, free_index=None, *, req=None, req_to_token_pool=None):
        if free_index is not None:
            super().free(free_index)
        if req is not None and req.kv.req_pool_idx is not None:
            self.get_kvcache().clear_request_scoped_state(int(req.kv.req_pool_idx))

    def _invalidate_ced(self, indices):
        coverage = self.get_kvcache()._dsv41_ced_coverage
        if coverage is not None:
            coverage.invalidate(indices)

    def clear_full_to_swa_mapping(self, full_indices):
        self._invalidate_ced(full_indices)
        return super().clear_full_to_swa_mapping(full_indices)

    def set_full_to_swa_mapping(self, full_indices, swa_indices):
        self._invalidate_ced(full_indices)
        return super().set_full_to_swa_mapping(full_indices, swa_indices)

    def free_full(self, free_index):
        self._invalidate_ced(free_index)
        return super().free_full(free_index)

    def clear(self):
        coverage = self.get_kvcache()._dsv41_ced_coverage
        if coverage is not None:
            coverage.swa.fill_(-1)
        return super().clear()


class RequestStateCleanup:
    _dsv41_allocator = None

    def register_dsv4_allocator(self, allocator):
        self._dsv41_allocator = allocator
        if getattr(allocator, "swa_req_ring", False):
            allocator._req_to_token_pool = self
            allocator._size_swa = self.available_size() * allocator.swa_ring_cost_tokens

    def free(self, req):
        if self._dsv41_allocator is not None:
            self._dsv41_allocator.free(req=req, req_to_token_pool=self)
        super().free(req)


class DSV41NPUReqToTokenPool(RequestStateCleanup, ReqToTokenPool):
    pass


class DSV41NPUDecodeReqToTokenPool(
    RequestStateCleanup, req_module.DecodeReqToTokenPool
):
    pass
