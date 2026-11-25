"""Merged replay media/embedding coordinates and the real API validation gate."""

import os
import unittest
from array import array
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import merged_prefill


def make_batch(requests, prefixes, counts, resets, device="cpu"):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=requests,
        prefix_lens=prefixes,
        extend_lens=counts,
        encoder_swa_reset=resets,
        req_pool_indices_cpu=torch.arange(1, len(requests) + 1),
        input_ids=torch.cat(
            [
                torch.tensor(
                    list(r.full_untruncated_fill_ids[p : p + n]), device=device
                )
                for r, p, n in zip(requests, prefixes, counts)
            ]
        ),
        out_cache_loc=torch.arange(sum(counts), device=device) + 128,
        extend_logprob_start_lens=list(counts),
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        multimodal_inputs=[r.multimodal_inputs for r in requests],
    )


def make_request(ids, media=None, dense=None, overrides=None):
    return SimpleNamespace(
        full_untruncated_fill_ids=array("q", ids),
        multimodal_inputs=media,
        input_embeds=dense,
        positional_embed_overrides=overrides,
    )


def make_runner(device="cpu"):
    return SimpleNamespace(
        device=device,
        token_to_kv_pool=SimpleNamespace(
            request_window=SimpleNamespace(valid_starts={1: 0, 2: 0, 3: 0})
        ),
        model=SimpleNamespace(model=SimpleNamespace(engram_hasher=None)),
    )


class ReplayInputs(unittest.TestCase):
    def test_dense_embeddings_and_overrides_follow_replay_coordinates(self):
        from sglang.srt.managers.embed_types import PositionalEmbeds

        dense = [torch.arange(n * 8).view(n, 8).float() for n in (320, 448)]
        overrides = [
            PositionalEmbeds(torch.full((4, 8), 11.0), [127, 128, 260, 320]),
            PositionalEmbeds(torch.full((3, 8), 22.0), [255, 256, 400]),
        ]
        requests = [
            make_request(range(len(d)), dense=d.tolist(), overrides=o)
            for d, o in zip(dense, overrides)
        ]
        batch = make_batch(requests, [256, 384], [64, 64], [True, True])
        batch.input_embeds = torch.cat([d[-64:] for d in dense])
        old_dense = batch.input_embeds
        view = merged_prefill(batch, make_runner())
        self.assertEqual(view.prefix_lens, [128, 256])
        self.assertEqual(view.extend_lens, [192, 192])
        torch.testing.assert_close(
            view.input_embeds, torch.cat([dense[0][128:], dense[1][256:]])
        )
        self.assertEqual(view.replace_positions.tolist(), [0, 132, 192, 336])
        torch.testing.assert_close(
            view.replace_embeds, torch.tensor([[11.0] * 8] * 2 + [[22.0] * 8] * 2)
        )
        self.assertIs(batch.input_embeds, old_dense)
        self.assertEqual(batch.prefix_lens, [256, 384])
        self.assertTrue(bool((view.out_cache_loc[:128] == -1).all()))
        self.assertTrue(bool((view.out_cache_loc[192:320] == -1).all()))

    def test_override_in_replay_only_and_later_request_offsets(self):
        from sglang.srt.managers.embed_types import PositionalEmbeds

        requests = [
            make_request(range(320)),
            make_request(
                range(320), overrides=PositionalEmbeds(torch.ones(1, 8), [130])
            ),
        ]
        batch = make_batch(requests, [256, 256], [64, 64], [True, True])
        self.assertIsNone(batch.replace_embeds)
        view = merged_prefill(batch, make_runner())
        self.assertEqual(view.replace_positions.tolist(), [194])
        self.assertEqual(view.replace_embeds.shape, (1, 8))
        self.assertIsNone(batch.replace_positions)

    def test_cold_continuation_keeps_prepared_inputs(self):
        request = make_request(range(320), media=object())
        for prefix, reset in ((0, True), (256, False)):
            batch = make_batch([request], [prefix], [64], [reset])
            batch.replace_embeds = torch.ones(1, 8)
            batch.replace_positions = torch.tensor([3])
            view = merged_prefill(batch, make_runner())
            self.assertIs(view.input_ids, batch.input_ids)
            self.assertIs(view.replace_embeds, batch.replace_embeds)
            self.assertIs(view.replace_positions, batch.replace_positions)

    def test_image_spans_match_full_prepared_embeddings(self):
        from sglang.srt.layers.engram import EngramHasher
        from sglang.srt.managers.req_types import MM_PAD_SHIFT_VALUE
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalInputs,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        devices = ["cpu"]
        if os.environ.get("SGLANG_DSV41_TEST_NPU") == "1":
            import torch_npu  # noqa: F401

            torch.npu.set_device(0)
            devices.append("npu:0")
        for device in devices:
            with (
                self.subTest(device=device),
                patch("sglang.srt.managers.mm_schedule._is_npu", device != "cpu"),
            ):
                requests = []
                for length, spans in (
                    (320, [(70, 219)]),
                    (448, [(10, 139), (240, 319)]),
                ):
                    ids = [i % 16 for i in range(length)]
                    items = []
                    for start, end in spans:
                        values = (
                            torch.arange((end - start + 1) * 8)
                            .view(-1, 8)
                            .to(device=device, dtype=torch.bfloat16)
                        )
                        item = MultimodalDataItem(
                            modality=Modality.IMAGE,
                            offsets=[(start, end)],
                            precomputed_embeddings=values,
                        )
                        item.set_identity((length * 1000 + start).to_bytes(32, "big"))
                        media_ids = item.padding_sequence(end - start + 1)
                        self.assertGreater(min(media_ids), torch.iinfo(torch.int32).max)
                        ids[start : end + 1] = media_ids
                        items.append(item)
                    requests.append(make_request(ids, MultimodalInputs(mm_items=items)))
                requests.append(make_request([i % 16 for i in range(134)]))
                embedding = torch.nn.Embedding(
                    16, 8, device=device, dtype=torch.bfloat16
                )
                model = SimpleNamespace(
                    vision=object(),
                    config=SimpleNamespace(image_token_id=15),
                    get_input_embeddings=lambda embedding=embedding: embedding,
                    get_image_feature=lambda items: self.fail(
                        "precomputed rows must be reused"
                    ),
                )
                model._prepare_mm_embeddings = MethodType(
                    DeepseekV4ForCausalLM._prepare_mm_embeddings, model
                )

                def prepare(ids, prefixes, counts, requests=requests, model=model):
                    fb = SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        mm_inputs=[r.multimodal_inputs for r in requests],
                        extend_prefix_lens_cpu=prefixes,
                        extend_seq_lens_cpu=counts,
                    )
                    return DeepseekV4ForCausalLM.prepare_language_model_inputs(
                        model, ids, fb
                    )

                full_ids = torch.cat(
                    [
                        torch.tensor(list(r.full_untruncated_fill_ids), device=device)
                        for r in requests
                    ]
                )
                full_tokens, full_embeds = prepare(full_ids, [0] * 3, [320, 448, 134])
                batch = make_batch(
                    requests, [256, 384, 128], [64, 64, 6], [True] * 3, device
                )
                source_ids = batch.input_ids.clone()
                runner = make_runner(device)
                hasher = SimpleNamespace(max_ngram_size=4)
                hasher.history_row = MethodType(EngramHasher.history_row, hasher)
                runner.model.model.engram_hasher = hasher
                view = merged_prefill(batch, runner)
                torch.testing.assert_close(
                    view.engram_history.cpu(),
                    torch.tensor(
                        [[MM_PAD_SHIFT_VALUE] * 3, [MM_PAD_SHIFT_VALUE] * 3, [0] * 3],
                        dtype=torch.int32,
                    ),
                )
                actual_tokens, actual_embeds = prepare(
                    view.input_ids, view.prefix_lens, view.extend_lens
                )
                expected_rows = torch.cat(
                    (
                        torch.arange(128, 320),
                        torch.arange(576, 768),
                        torch.arange(768, 902),
                    )
                )
                torch.testing.assert_close(
                    actual_embeds.cpu(),
                    full_embeds.cpu()[expected_rows],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual_tokens.cpu(),
                    full_tokens.cpu()[expected_rows],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(batch.input_ids, source_ids, rtol=0, atol=0)
                self.assertIs(view.multimodal_inputs, batch.multimodal_inputs)
                self.assertIs(merged_prefill(view, runner), view)


def test_running_radix_checkpoint_keeps_request_owned_draft_window():
    """An existing FULL prefix may lack draft SWA for a shorter request's tail."""
    from sglang.srt.hardware_backend.npu.dsv41.prefill.cache import adapt_checkpoint
    from sglang.srt.managers.schedule_batch import ReqKvInfo
    from sglang.srt.mem_cache.common import checkpoint_kv_cache

    page = 128
    own_slots = torch.arange(1024) + 128
    tree_slots = own_slots + 1024
    req_slots = own_slots.clone()
    pool = SimpleNamespace(
        _dsv41_910b=True, request_window=object(), needs_paged_swa_allocator=True
    )
    tree = SimpleNamespace(
        page_size=page,
        disable=False,
        decoder_swa_bounded_replay=True,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
        advance_unpublished_req=lambda req: None,
    )

    def adopt_tree_slots(cache, req, *, up_to):
        boundary = up_to // cache.page_size * cache.page_size
        req_slots[req.kv.cache_protected_len : boundary] = tree_slots[
            req.kv.cache_protected_len : boundary
        ]
        req.kv.protect_prefix(boundary)

    req = SimpleNamespace(
        origin_input_ids=range(1024),
        skip_radix_cache_insert=False,
        prefix_len=0,
        extend_end=512,
        kv=ReqKvInfo(req_pool_idx=1),
        finished=lambda: False,
    )
    with patch.dict(os.environ, {"SGLANG_DSV41_910B": "1"}):
        tree.checkpoint = MethodType(adapt_checkpoint(adopt_tree_slots), tree)

    checkpoint_kv_cache(req, tree)
    assert req.prefix_len == req.kv.cache_protected_len == 512
    req.extend_end = 1024
    checkpoint_kv_cache(req, tree)

    assert req.prefix_len == 1024
    assert req.kv.cache_protected_len == 768
    torch.testing.assert_close(req_slots[:768], tree_slots[:768])
    torch.testing.assert_close(req_slots[768:], own_slots[768:])

    req.finished = lambda: True
    tree.checkpoint(req, up_to=1024)
    torch.testing.assert_close(req_slots, tree_slots)


class TokenizerValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="/mnt/models/DeepSeek-V4.1-Flash-1017",
            device="npu",
            dtype="bfloat16",
            kv_cache_dtype="bfloat16",
            enable_encoder_swa_bounded_replay=True,
        )
        # Argument validation does not need a live NPU or a memory query.
        with patch(
            "sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity",
            return_value=65536,
        ):
            publish(args, role="tokenizer")
        manager = object.__new__(TokenizerManager)
        manager.server_args = args
        manager.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            joint_head_config=None,
            pointer_head=None,
        )
        manager.context_len, manager.num_reserved_tokens = 4096, 0
        manager.allow_auto_truncate, manager.validate_total_tokens = False, True
        manager.is_generation = True
        cls.manager = manager

    def test_media_and_embeddings_keep_length_and_logprob_validation(self):
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.runtime_context import get_exec

        for field, value in (
            ("image_data", ["image"]),
            ("video_data", ["video"]),
            ("audio_data", ["audio"]),
            ("input_embeds", [[0.0] * 8]),
            ("positional_embed_overrides", object()),
        ):
            with self.subTest(field=field):
                obj = GenerateReqInput(
                    input_ids=[1] * 256, sampling_params={"max_new_tokens": 1}
                )
                setattr(obj, field, value)
                self.manager._validate_one_request(obj, obj.input_ids)
                self.assertTrue(get_exec().features.enable_encoder_swa_bounded_replay)
                obj.return_logprob, obj.logprob_start_len = True, 128
                with self.assertRaisesRegex(ValueError, "cached prompt logprobs"):
                    self.manager._validate_one_request(obj, obj.input_ids)
                obj.logprob_start_len = 256
                self.manager._validate_one_request(obj, obj.input_ids)
                obj.logprob_start_len = -1
                with self.assertRaisesRegex(ValueError, "longer than"):
                    self.manager._validate_one_request(obj, [1] * 4096)
                self.assertTrue(get_exec().features.enable_encoder_swa_bounded_replay)

    def test_other_models_retain_the_upstream_gate(self):
        from sglang.srt.managers.io_struct import GenerateReqInput

        original = self.manager.model_config.hf_config.model_type
        self.manager.model_config.hf_config.model_type = "another_model"
        try:
            obj = GenerateReqInput(input_ids=[1] * 256, image_data=["image"])
            with self.assertRaisesRegex(ValueError, "encoder SWA replay"):
                self.manager._validate_one_request(obj, obj.input_ids)
        finally:
            self.manager.model_config.hf_config.model_type = original


if __name__ == "__main__":
    unittest.main()
