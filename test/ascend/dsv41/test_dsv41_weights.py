"""Streaming BF16 loader contracts and opt-in real checkpoint/NPU coverage.

Run with SGLANG_DSV41_910B=1 and PYTHONPATH=python.
NPU: set SGLANG_DSV41_TEST_NPU=1 SGLANG_DSV41_910B=1 and a physical device mapping.
"""

import importlib.util
import json
import os
from pathlib import Path
import unittest
import tempfile
import time
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.dsv41.dense_weights import prepare_weights


def reference_decode(weight, scale):
    """Independent checkpoint definition, including partial edge blocks."""
    rows = torch.arange(weight.shape[0]) // 32
    cols = torch.arange(weight.shape[1]) // 32
    return (weight.float() * scale.float()[rows[:, None], cols[None, :]]).bfloat16()


class StreamingWeightsTest(unittest.TestCase):
    def setUp(self):
        self.weight = (torch.arange(35 * 65).reshape(35, 65) % 31 - 15).to(
            torch.float8_e4m3fn
        )
        self.scale = torch.tensor(
            [[126, 127, 128], [123, 124, 125]], dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)

    def test_pair_order_and_interleaved_passthrough(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                pair = [
                    ("layers.0.attn.wq_a.weight", self.weight),
                    ("layers.0.attn.wq_a.scale", self.scale),
                ]
                if reverse:
                    pair.reverse()
                norm = torch.ones(35, dtype=torch.bfloat16)
                result = dict(
                    prepare_weights(
                        [pair[0], ("norm.weight", norm), pair[1]],
                        decoder=reference_decode,
                    )
                )
                self.assertEqual(
                    set(result), {"norm.weight", "layers.0.attn.wq_a.weight"}
                )
                self.assertIs(result["norm.weight"], norm)
                torch.testing.assert_close(
                    result["layers.0.attn.wq_a.weight"],
                    reference_decode(self.weight, self.scale),
                    rtol=0,
                    atol=0,
                )

    def test_expert_and_engram_tables_retain_compressed_storage(self):
        for prefix in ("layers.1.ffn.experts.0.w1", "layers.1.engram.embed"):
            result = dict(
                prepare_weights(
                    [
                        (prefix + ".weight", self.weight),
                        (prefix + ".scale", self.scale),
                    ],
                    decoder=lambda *_: self.fail("host compressed payload was decoded"),
                )
            )
            self.assertIs(result[prefix + ".weight"], self.weight)
            self.assertIs(result[prefix + ".scale"], self.scale)

    def test_shared_engram_projection_and_scale_inv(self):
        for prefix in ("layers.1.ffn.shared_experts.w1", "layers.1.engram.wkv"):
            result = dict(
                prepare_weights(
                    [
                        (prefix + ".weight_scale_inv", self.scale),
                        (prefix + ".weight", self.weight),
                    ],
                    decoder=reference_decode,
                )
            )
            self.assertEqual(list(result), [prefix + ".weight"])
            self.assertEqual(result[prefix + ".weight"].dtype, torch.bfloat16)

    def test_missing_or_duplicate_pair_fails(self):
        for tensors in (
            [("a.weight", self.weight)],
            [("a.scale", self.scale)],
            [("a.weight", self.weight), ("a.weight", self.weight)],
        ):
            with self.subTest(tensors=[name for name, _ in tensors]):
                with self.assertRaises(ValueError):
                    list(prepare_weights(tensors, decoder=reference_decode))

    def test_wrong_scale_geometry_fails(self):
        with self.assertRaisesRegex(ValueError, "32x32"):
            list(
                prepare_weights(
                    [("a.weight", self.weight), ("a.scale", self.scale[:1])],
                    decoder=reference_decode,
                )
            )

    def test_indexed_reader_rejects_incomplete_expert_before_reading(self):
        from sglang.srt.hardware_backend.npu.dsv41.checkpoint import (
            ExpertCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layers.0.ffn.experts.0.w1.weight": "absent.safetensors"
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "incomplete V4.1 expert"):
                ExpertCheckpoint(root)

    def test_indexed_reader_filters_target_and_draft_before_reading(self):
        from sglang.srt.hardware_backend.npu.dsv41.checkpoint import (
            ExpertCheckpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = {
                prefix + "." + projection + "." + kind: "absent.safetensors"
                for prefix in ("layers.0.ffn.experts.0", "mtp.0.ffn.experts.0")
                for projection in ("w1", "w2", "w3")
                for kind in ("weight", "scale")
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": index})
            )
            self.assertEqual(
                set(ExpertCheckpoint(root).groups), {"layers.0.ffn.experts.0"}
            )
            self.assertEqual(
                set(ExpertCheckpoint(root, draft=True).groups), {"mtp.0.ffn.experts.0"}
            )


@unittest.skipUnless(
    importlib.util.find_spec("acl"),
    "requires the CANN Python runtime for SGLang imports",
)
class LinearPolicyTest(unittest.TestCase):
    def test_w8a16_covers_dense_shared_and_draft_linears(self):
        from sglang.srt.hardware_backend.npu.dsv41.dense_weights import adapt_get_method
        from sglang.srt.hardware_backend.npu.dsv41.linear import W8A16LinearMethod
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        with patch.dict(os.environ, SGLANG_DSV41_910B="1"):
            get_method = adapt_get_method(Fp8Config.get_quant_method)
            config = Fp8Config(
                True, "dynamic", weight_block_size=[32, 32], scale_fmt="ue8m0"
            )
            config._dsv41_bf16_dense = True
            layer = LinearBase.__new__(LinearBase)
            for prefix in (
                "model.layers.0.self_attn.wq_a",
                "model.layers.0.mlp.shared_experts.gate_up_proj",
                "stages.0.main_proj",
                "stages.0.self_attn.wo_a",
            ):
                self.assertIsInstance(get_method(config, layer, prefix), W8A16LinearMethod)


@unittest.skipUnless(
    os.environ.get("SGLANG_DSV41_TEST_NPU") == "1",
    "set SGLANG_DSV41_TEST_NPU=1 for real checkpoint/NPU validation",
)
class CheckpointNPUTest(unittest.TestCase):
    def test_image_preprocessing_matches_checkpoint(self):
        import importlib.util
        import io
        import sys
        from PIL import Image
        from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
        from sglang.srt.multimodal.deepseek_v41_image_processing import (
            patchify_image,
            image_token_types,
        )

        root = Path(
            os.environ.get("SGLANG_DSV41_MODEL", "/mnt/models/DeepSeek-V4.1-Flash-1017")
        )
        config = DeepseekV41Config(**json.loads((root / "config.json").read_text()))
        spec = importlib.util.spec_from_file_location(
            "checkpoint_image_reference", root / "inference/image_processor.py"
        )
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
        for width, height in ((42, 84), (32, 47), (144, 233), (2048, 32)):
            with self.subTest(size=(width, height)):
                pixels = torch.randint(
                    256, (height, width, 3), dtype=torch.uint8
                ).numpy()
                image = Image.fromarray(pixels)
                encoded = io.BytesIO()
                image.save(encoded, format="PNG")
                expected = ref.load_image({"data": encoded.getvalue()}, config)
                actual = patchify_image(image, config)
                self.assertEqual(actual[1:], expected[1:])
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                self.assertTrue(
                    torch.equal(
                        image_token_types(*actual[3:]),
                        ref.image_token_types(*expected[3:]),
                    )
                )
                self.assertLessEqual(
                    len(image_token_types(*actual[3:])), config.vision_max_n_token
                )

    def test_real_vision_aligner_features(self):
        """32 real ViT layers, CPU checkpoint reference, non-square/pad grids."""
        import importlib.util
        from types import MethodType, SimpleNamespace

        import torch_npu  # noqa: F401
        from safetensors import safe_open
        from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
        from sglang.srt.runtime_context import get_parallel, publish
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.models.deepseek_v41_vit import ViT, Aligner

        root = Path(
            os.environ.get("SGLANG_DSV41_MODEL", "/mnt/models/DeepSeek-V4.1-Flash-1017")
        )
        config = DeepseekV41Config(**json.loads((root / "config.json").read_text()))
        publish(
            ServerArgs(
                model_path=str(root),
                device="npu",
                dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                mm_attention_backend="ascend_attn",
            ),
            role="scheduler",
        )
        args = SimpleNamespace(**vars(config), dim=config.hidden_size)
        spec = importlib.util.spec_from_file_location(
            "checkpoint_vision_reference", root / "inference/vision.py"
        )
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        torch.npu.set_device(0)
        torch.set_num_threads(8)
        dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                tp_group=None,
                attn_tp_group=None,
            ):
                with torch.device("npu"):
                    vision, aligner = ViT(args), Aligner(args)
            with torch.device("cpu"):
                expected_vision, expected_aligner = ref.ViT(args), ref.Aligner(args)
        finally:
            torch.set_default_dtype(dtype)
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for prefix, actual, expected in (
            ("vision", vision, expected_vision),
            ("aligner", aligner, expected_aligner),
        ):
            params = dict(actual.named_parameters())
            loaded = set()
            for local, param in expected.named_parameters():
                name = prefix + "." + local
                with safe_open(
                    root / index[name], framework="pt", device="cpu"
                ) as source:
                    value = source.get_tensor(name)
                self.assertIn(value.dtype, (torch.bfloat16, torch.float32))
                mapped = local.replace(".attn.wqkv.", ".attn.qkv_proj.").replace(
                    ".attn.wo.", ".attn.proj."
                )
                self.assertEqual(value.shape, params[mapped].shape)
                if params[mapped].dtype != value.dtype:
                    self.assertEqual(params[mapped].dtype, torch.float32)
                    self.assertIn("norm", mapped)
                params[mapped].data.copy_(value)
                param.data.copy_(value)
                loaded.add(mapped)
            self.assertEqual(set(params), loaded, f"unloaded {prefix} parameters")

        # Kunpeng's CPU BF16 linear fallback is scalar. Compute the reference
        # dot products in FP32 with the exact BF16 checkpoint values, then
        # round at every original BF16 projection handoff. Keep the checkpoint
        # RoPE, RMS, SiLU/GELU, padding and unfold formulas independent.
        def cpu_linear(module, value):
            return torch.nn.functional.linear(
                value.float(), module.weight, module.bias
            ).to(value.dtype)

        for reference_model in (expected_vision, expected_aligner):
            for module in reference_model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.float()
                    module.forward = MethodType(cpu_linear, module)

        def cpu_attention(module, value, cos, sin):
            n = value.shape[0]
            q, k, v = (
                t.view(n, module.n_heads, module.head_dim)
                for t in module.wqkv(value).chunk(3, -1)
            )
            q, k = ref.apply_rotary(q, cos, sin), ref.apply_rotary(k, cos, sin)
            output = torch.nn.functional.scaled_dot_product_attention(
                q.float().transpose(0, 1),
                k.float().transpose(0, 1),
                v.float().transpose(0, 1),
            )
            return module.wo(output.transpose(0, 1).reshape(n, -1).to(value.dtype))

        for block in expected_vision.blocks:
            block.attn.forward = MethodType(cpu_attention, block.attn)
        torch.manual_seed(1072)
        with (
            torch.inference_mode(),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                tp_group=None,
                attn_tp_group=None,
            ),
        ):
            for h, w in ((9, 9), (7, 11), (14, 15), (39, 39), (93, 93)):
                with self.subTest(grid=(h, w)):
                    patches = torch.randn(h * w, 3, 14, 14, dtype=torch.bfloat16)
                    expected_features = expected_vision(patches, h, w)
                    expected_rows = expected_aligner(expected_features, h, w)
                    device_patches = patches.npu()
                    torch.npu.synchronize()
                    started = time.perf_counter()
                    features = vision(device_patches, h, w)
                    rows = aligner(features, h, w)
                    torch.npu.synchronize()
                    print(
                        f"VISION encoding grid={h}x{w} first_call_seconds={time.perf_counter() - started:.6f}",
                        flush=True,
                    )
                    for label, a, b in (
                        ("vit", features.cpu(), expected_features),
                        ("aligner", rows.cpu(), expected_rows),
                    ):
                        # Large FP32 ARM cosine reductions can exceed one;
                        # accumulate validation metrics independently in FP64.
                        a, b = a.double(), b.double()
                        relative = float((a - b).norm() / b.norm())
                        cosine = float(
                            torch.nn.functional.cosine_similarity(
                                a.flatten(), b.flatten(), dim=0
                            )
                        )
                        self.assertLess(relative, 0.04, (label, h, w, relative, cosine))
                        self.assertGreater(
                            cosine, 0.999, (label, h, w, relative, cosine)
                        )
                        self.assertLessEqual(cosine, 1 + 1e-12)
                        print(
                            f"VISION {label} grid={h}x{w} relative_l2={relative:.6g} cosine={cosine:.8f}",
                            flush=True,
                        )

    def test_real_dense_shared_and_engram_projection_graph(self):
        import torch_npu  # noqa: F401
        from safetensors import safe_open

        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        torch.npu.set_device(0)
        directory = Path(
            os.environ.get("SGLANG_DSV41_MODEL", "/mnt/models/DeepSeek-V4.1-Flash-1017")
        )
        index = json.loads((directory / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        # Includes the special Engram projection shape [25600,6144].
        for prefix in (
            "layers.0.attn.wq_a",
            "layers.0.ffn.shared_experts.w1",
            "layers.1.engram.wkv",
        ):
            with self.subTest(prefix=prefix):
                pair = []
                for suffix in (".scale", ".weight"):
                    key = prefix + suffix
                    with safe_open(
                        directory / index[key], framework="pt", device="cpu"
                    ) as source:
                        pair.append((key, source.get_tensor(key)))
                weight, scale = pair[1][1], pair[0][1]
                decoded = dict(prepare_weights(pair))[prefix + ".weight"]
                # Every element is checked; generate the FP32 reference in
                # bounded CPU chunks, including every block boundary.
                for start in range(0, weight.shape[0], 128):
                    stop = min(start + 128, weight.shape[0])
                    expected = reference_decode(
                        weight[start:stop], scale[start // 32 : (stop + 31) // 32]
                    )
                    torch.testing.assert_close(
                        decoded[start:stop].cpu(), expected, rtol=0, atol=0
                    )
                config = Fp8Config(
                    True, "dynamic", weight_block_size=[32, 32], scale_fmt="ue8m0"
                )
                config._dsv41_bf16_dense = True
                with torch.device("npu:0"):
                    layer = ReplicatedLinear(
                        weight.shape[1],
                        weight.shape[0],
                        bias=False,
                        params_dtype=torch.bfloat16,
                        quant_config=config,
                        prefix=prefix,
                    )
                self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
                self.assertFalse(hasattr(layer, "weight_scale_inv"))
                layer.weight.weight_loader(layer.weight, decoded)
                del decoded
                for tokens in (1, 6):
                    x = torch.randn(
                        tokens, weight.shape[1], dtype=torch.bfloat16, device="npu:0"
                    )
                    stream = torch.npu.Stream()
                    with torch.npu.stream(stream):
                        layer(x)
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        output = layer(x)[0]
                    for factor in (0.5, 0.25):
                        x.mul_(factor)
                        torch.npu.synchronize()
                        graph.replay()
                        torch.npu.synchronize()
                        expected = torch.nn.functional.linear(x, layer.weight)
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                del layer, pair, weight, scale
                torch.npu.synchronize()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
