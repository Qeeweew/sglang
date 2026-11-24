"""Native V4.1 DSpark checkpoint and noncausal BF16 draft attention on 910B."""

from __future__ import annotations

import copy
import logging
from functools import wraps
from types import SimpleNamespace

import torch
from sglang.srt.hardware_backend.npu.dsv41.dense_weights import enabled, prepare_weights
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from .. import group_cp2_dp2

logger = logging.getLogger(__name__)


class DSV41DraftBackend(AttentionBackend):
    """One stable BF16 draft KV pool; full-block visibility after all KV writes."""

    data_type = kv_cache_dtype = torch.bfloat16
    supports_ragged_verify_graph = False
    needs_cpu_seq_lens = False
    attn_backend_list = None
    forward_metadata = None
    kv_index_translator = None

    def __init__(self, runner):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import PositionQuant

        self.runner = runner
        self.pool = runner.token_to_kv_pool
        self.req_pool = runner.req_to_token_pool
        self.token_to_kv_pool, self.req_to_token_pool = self.pool, self.req_pool
        self.device = torch.device("npu", torch.npu.current_device())
        self.position = PositionQuant(device=self.device)
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import DraftPageMetadata
        from sglang.srt.runtime_context import get_parallel

        self.page_metadata = DraftPageMetadata(device=self.device)
        attn_tp = 1 if group_cp2_dp2.enabled() else get_parallel().attn_tp_size
        self.local_heads = runner.model_config.hf_config.num_attention_heads // attn_tp
        self.capacity = runner.server_args.max_running_requests
        self.window_capacity = 256
        table_capacity = self.req_pool.req_to_token.shape[1]
        self.max_seq = ((table_capacity + 127) // 128) * 128
        self.buckets = {}

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 5

    def shared_read_ends(self, fm):
        from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds

        return SharedReadEnds.IN_REPLAY

    def init_forward_metadata(self, fb):
        self.init_forward_metadata_out_graph(fb)
        self.init_forward_metadata_in_graph(fb)

    def init_forward_metadata_in_graph(self, fb):
        if fb.forward_mode.is_idle():
            return
        meta = self.forward_metadata
        self._prepare_pages(meta)
        local = meta.group_queries
        if local is not None:
            shape = meta.group_decode
            for field, per_request, pad in (
                ("requests", True, 0),
                ("prefix", True, 0),
                ("full_slots", False, -1),
            ):
                getattr(local, field).copy_(
                    shape.slice(
                        getattr(meta, field),
                        role=("draft_pages", field),
                        per_request=per_request,
                        pad_value=pad,
                    )
                )
            self._prepare_pages(local)

    def _prepare_pages(self, meta):
        batch, block = meta.batch, meta.block
        requests = meta.requests[:batch]
        self.page_metadata(
            self.req_pool.req_to_token,
            self.pool.full_to_swa_index_mapping,
            requests,
            meta.prefix,
            meta.full_slots,
            cu=meta.cu,
            lengths=meta.lengths,
            table=meta.table,
            slots=meta.slots,
            routed_rows=meta.routed_rows,
            block=block,
        )

    def _bucket(self, batch, rows, *, local=False):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import DraftAttention

        key = (batch, rows, local)
        if key not in self.buckets:

            def empty(shape, dtype):
                return torch.zeros(shape, dtype=dtype, device=self.device)

            self.buckets[key] = SimpleNamespace(
                cu=empty((batch + 1,), torch.int32),
                lengths=empty((batch,), torch.int32),
                table=empty((batch, self.max_seq // 128), torch.int32),
                slots=empty((rows,), torch.int64),
                full_slots=empty((rows,), torch.int64),
                requests=empty((batch,), torch.int64),
                prefix=empty((batch,), torch.int64),
                routed_rows=empty((1,), torch.int32),
                attention=DraftAttention(
                    capacity=self.window_capacity,
                    heads=self.local_heads,
                ),
            )
        return self.buckets[key]

    def init_forward_metadata_out_graph(self, fb, in_capture=False):
        if fb.forward_mode.is_idle():
            return
        batch, rows = fb.batch_size, fb.input_ids.numel()
        if batch > self.capacity:
            raise RuntimeError(
                f"draft batch {batch} exceeds max_running_requests={self.capacity}"
            )
        grouped = group_cp2_dp2.decode_step(fb)
        if batch <= 0 or (not grouped and rows % batch):
            raise ValueError("DSpark draft rows must form complete per-request blocks")
        block = fb.spec_info.num_tokens_per_req if grouped else rows // batch
        if grouped:
            # Eager MLP alignment can append one token outside the request blocks.
            if rows not in (batch * block, batch * block + 1):
                raise ValueError("invalid padding outside DSpark draft request blocks")
            rows = batch * block
        if block <= 0 or block + 128 > self.window_capacity:
            raise RuntimeError("DSpark block does not fit the 256-token draft window")
        # build_replay_fb_view combines padded query/request buffers with the
        # original, potentially shorter cache locations and CPU length mirror.
        raw_batch = batch - int(getattr(fb, "num_padding", 0))
        if not 0 <= raw_batch <= batch:
            raise ValueError("invalid DSpark draft replay padding")
        real_rows = raw_batch * block
        if real_rows and (
            fb.out_cache_loc is None or fb.out_cache_loc.numel() < real_rows
        ):
            raise ValueError(
                "DSpark draft cache locations do not cover real query rows"
            )
        meta = self._bucket(batch, rows)
        meta.batch, meta.block, meta.rows = batch, block, rows
        meta.requests.zero_()
        meta.prefix.zero_()
        meta.full_slots.fill_(-1)
        if raw_batch:
            meta.requests[:raw_batch].copy_(fb.req_pool_indices[:raw_batch])
            meta.prefix[:raw_batch].copy_(
                fb.positions[: batch * block].view(batch, block)[:raw_batch, 0]
            )
            meta.full_slots[:real_rows].copy_(fb.out_cache_loc[:real_rows])
        meta.group_queries = None
        if grouped:
            shape = group_cp2_dp2.DecodeRows(batch, block)
            local = self._bucket(shape.local_batch, shape.local_rows, local=True)
            local.batch, local.block = shape.local_batch, block
            if not hasattr(local, "swa_pending"):
                local.swa_pending = torch.empty(
                    (len(self.runner.model.stages), shape.local_rows, 512),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            meta.group_decode, meta.group_queries = shape, local
        self.forward_metadata = meta

    def get_swa_out_cache_loc(self, forward_batch):
        meta = self.forward_metadata
        return (meta.group_queries or meta).slots

    def store_cache(self, *, layer_id, swa_k, forward_batch):
        meta = self.forward_metadata
        local = meta.group_queries
        if local is not None:
            local.swa_pending[self.pool._swa_local_layer_id(layer_id)].copy_(swa_k)
        self.position.scatter_bf16(
            swa_k.contiguous(),
            (local or meta).slots,
            self.pool.get_swa_buffer(layer_id).squeeze(2),
        )

    def forward(self, q, k, v, layer, forward_batch, *, attn_sink, **kwargs):
        meta = self.forward_metadata.group_queries or self.forward_metadata
        return meta.attention(
            q,
            self.pool.get_swa_buffer(layer.layer_id),
            meta.table,
            meta.cu,
            meta.lengths,
            attn_sink.contiguous(),
        )

    def mirror_group_swa(self, stages):
        meta = self.forward_metadata
        shape = meta.group_decode
        pending = shape.gather_layers(
            meta.group_queries.swa_pending, role="draft_swa", skip_own_output=True
        )
        peer = shape.peer_slice
        slots = meta.slots[peer]
        if not slots.numel():
            return
        for stage in stages:
            layer = stage.self_attn.layer_id
            self.position.scatter_bf16(
                pending[self.pool._swa_local_layer_id(layer), peer],
                slots,
                self.pool.get_swa_buffer(layer).squeeze(2),
            )


def adapt_model_forward(original):
    if not enabled():
        return original

    @wraps(original)
    def forward(model, input_ids, positions, batch, input_embeds):
        if input_ids.device.type != "npu" or not group_cp2_dp2.decode_step(batch):
            return original(model, input_ids, positions, batch, input_embeds)
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        if not isinstance(backend, DSV41DraftBackend):
            return original(model, input_ids, positions, batch, input_embeds)
        idle = batch.forward_mode.is_idle()
        shape = (
            group_cp2_dp2.DecodeRows(
                batch.batch_size, batch.spec_info.num_tokens_per_req
            )
            if idle
            else backend.forward_metadata.group_decode
        )
        ids = shape.slice(input_ids, role="draft_ids")
        pos = shape.slice(positions, role="draft_positions")
        embeds = (
            None
            if input_embeds is None
            else shape.slice(input_embeds, role="draft_embeds")
        )
        group_cp2_dp2.prepare_moe_rows(batch, ids, rows_per_request=shape.width)
        result = original(model, ids, pos, batch, embeds)
        if not idle:
            backend.mirror_group_swa(model.stages)
        result.hidden_states = shape.gather(result.hidden_states, role="draft_output")
        return result

    return forward


def load_draft_weights(model, weights):
    """Read native mtp.* only, decoding dense pairs and preserving raw experts."""
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.model_loader.weight_utils import default_weight_loader
    from sglang.srt.models.deepseek_v4 import DEEPSEEK_V4_STACKED_PARAMS_MAPPING

    params = dict(model.named_parameters())
    loaded = set()
    experts = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=int(model.config.dspark_n_routed_experts),
    )
    # Filtering precedes decode: never convert the target's hundreds of dense
    # matrices just to discard them while loading the three draft stages.
    selected = ((name, value) for name, value in weights if name.startswith("mtp."))
    for name, value in prepare_weights(selected):
        mapped = model._remap_dspark_weight_name(name)
        if mapped is None:
            continue
        for target, source, shard in DEEPSEEK_V4_STACKED_PARAMS_MAPPING:
            candidate = mapped.replace(source, target)
            if source in mapped and candidate in params:
                param = params[candidate]
                param.weight_loader(param, value, shard)
                loaded.add(candidate)
                break
        else:
            for target, source, expert, shard in experts:
                candidate = mapped.replace(source, target)
                if source in mapped and candidate in params:
                    param = params[candidate]
                    param.weight_loader(
                        param, value, candidate, expert_id=expert, shard_id=shard
                    )
                    loaded.add(candidate)
                    break
            else:
                if mapped not in params:
                    raise ValueError(
                        f"unexpected native V4.1 DSpark weight {name} -> {mapped}"
                    )
                param = params[mapped]
                getattr(param, "weight_loader", default_weight_loader)(param, value)
                loaded.add(mapped)
    model._assert_confidence_head_loaded(params_dict=params, loaded_params=loaded)
    # The retained FP8 quant-config installs legacy RadixAttention scales.
    # BF16 draft KV never consumes them, and the native checkpoint has none.
    missing = {
        name
        for name in set(params) - loaded
        if not name.endswith((".attn.k_scale", ".attn.v_scale"))
    }
    if missing:
        raise ValueError(f"missing V4.1 DSpark weights: {sorted(missing)[:12]}")
    for stage in model.stages:
        stage.refresh_mhc_norm_weight_cache()
    from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
        compact_rope_storage,
    )

    compact_rope_storage(model)
    logger.debug(
        "Loaded %d native V4.1 DSpark parameters, 3 stages, E128/top3", len(loaded)
    )


def adapt_draft_config(original_config):
    if not enabled():
        return original_config

    @wraps(original_config)
    def draft_config(config):
        result = original_config(config)
        hf = config.hf_config
        if (
            getattr(hf, "model_type", None) == "deepseek_v41"
            and hf.architectures[0] == "DeepseekV4ForCausalLMDSpark"
        ):
            previous = hf
            hf = copy.copy(hf)
            config.hf_config = hf
            if config.hf_text_config is previous:
                config.hf_text_config = hf
            count = int(hf.num_nextn_predict_layers)
            hf.num_hidden_layers = count
            hf.compress_ratios = [0] * count
            hf.kv_source_layer_ids = []
            hf.index_source_layer_ids = []
            hf.candidate_source_layer_id = -1
            hf.engram_layer_ids = []
            hf.n_routed_experts = int(hf.dspark_n_routed_experts)
            hf.num_experts_per_tok = int(hf.dspark_num_experts_per_tok)
        return result

    return draft_config


def adapt_load(original_load):
    if not enabled():
        return original_load

    @wraps(original_load)
    def load(model, weights):
        if getattr(model.config, "model_type", None) == "deepseek_v41":
            return load_draft_weights(model, weights)
        return original_load(model, weights)

    return load


def adapt_inject(original_inject):
    if not enabled():
        return original_inject

    def rotate(self, x, positions, *, out=None):
        from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
            position_operator,
        )

        op = position_operator(x.device)
        from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
            gather_rope,
        )

        trig = gather_rope(self.freqs_cis, positions.long())
        return op.rope(x.contiguous(), trig.cos, trig.sin, out=out)

    @wraps(original_inject)
    def inject(model, *, main_hidden, swa_loc, positions, pool):
        from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
            position_operator,
        )

        if getattr(model.config, "model_type", None) != "deepseek_v41":
            return original_inject(
                model,
                main_hidden=main_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )
        if (
            main_hidden.shape[0] != swa_loc.numel()
            or positions.numel() != swa_loc.numel()
        ):
            raise ValueError(
                "V4.1 draft injection requires aligned hidden, positions and SWA locations"
            )
        shape = None
        if group_cp2_dp2.enabled():
            rows = main_hidden.shape[0]
            if not rows:
                return
            # Prefill and verify supply the same DP-local rows to both TP lanes.
            shape = group_cp2_dp2.DecodeRows(rows, 1)
            main_hidden = shape.slice(main_hidden, role="inject_hidden")
            positions = shape.slice(positions, role="inject_positions")

        main = model.project_target_hidden(main_hidden)
        op = position_operator(main.device)
        pending = (
            None
            if shape is None
            else main.new_empty((len(model.stages), shape.local_rows, 512))
        )
        slots = swa_loc.long()
        for i, stage in enumerate(model.stages):
            layer = stage.self_attn
            kv = rotate(
                layer,
                layer.kv_norm(layer.wkv(main)[0]),
                positions,
                out=None if pending is None else pending[i],
            )
            if pending is None:
                op.scatter_bf16(
                    kv, slots, pool.get_swa_buffer(layer.layer_id).squeeze(2)
                )
        if shape is not None:
            gathered = shape.gather_layers(
                pending, role="draft_inject", comm_role="inject"
            )
            for stage, kv in zip(model.stages, gathered):
                layer = stage.self_attn
                op.scatter_bf16(
                    kv, slots, pool.get_swa_buffer(layer.layer_id).squeeze(2)
                )

    return inject


def adapt_attention_init(original_attention_init):
    if not enabled():
        return original_attention_init

    @wraps(original_attention_init)
    def attention_init(self, *args, **kwargs):
        original_attention_init(self, *args, **kwargs)
        if self.is_dsv41 and self.wo_b.parallel_group == "attn_tp":
            self.wo_b.use_dp_attention_reduce = True

    return attention_init


def adapt_forward(original_attention):
    if not enabled():
        return original_attention

    def rotate(self, x, positions, *, inverse=False):
        from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
            position_operator,
        )

        op = position_operator(x.device)
        from sglang.srt.hardware_backend.npu.dsv41.attention.projections import gather_rope

        trig = gather_rope(self.freqs_cis, positions.long())
        return op.rope(x.contiguous(), trig.cos, trig.sin, inverse=inverse)

    @wraps(original_attention)
    def forward(
        self, positions, hidden_states, forward_batch, *, defer_all_reduce=False
    ):
        if not self.is_dsv41 or hidden_states.device.type != "npu":
            return original_attention(
                self,
                positions,
                hidden_states,
                forward_batch,
                defer_all_reduce=defer_all_reduce,
            )
        import torch_npu
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul

        if defer_all_reduce:
            raise ValueError("910B draft attention requires its ordinary TP reduction")
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(hidden_states)
        backend = get_attn_backend()
        kv, _ = self.wkv(hidden_states)
        kv = rotate(self, self.kv_norm(kv), positions)
        backend.store_cache(
            layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch
        )
        low, _ = self.wq_a(hidden_states)
        low = self.q_norm(low)
        q, _ = self.wq_b(low)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.q_head_norm:
            q = torch_npu.npu_rms_norm(q, self._q_post_norm_weight, self.eps)[0]
        q = rotate(self, q, positions)
        o = backend.forward(
            q,
            kv,
            kv,
            self.attn,
            forward_batch,
            attn_sink=self._local_attn_sink(kernel_num_heads=self.n_local_heads),
            save_kv_cache=False,
        )
        o = rotate(self, o, positions, inverse=True)
        grouped = o.view(o.shape[0], self.n_local_groups, -1)
        weight = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = _apply_wo_a_bf16_matmul(
            grouped,
            weight,
            is_decode=forward_batch.forward_mode.is_decode(),
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
        )
        return self.wo_b(o.reshape(o.shape[0], -1))[0]

    return forward
