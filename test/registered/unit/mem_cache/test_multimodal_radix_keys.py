import sys
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import _req_radix_key
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.utils import get_hash_str, get_storage_hash_str
from sglang.srt.multimodal.modality import Modality
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import slot_of, tree_slots_of

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _request(tokens, media, *, salt="tenant"):
    tokens = array("q", tokens)
    items = []
    for offset, identity in media:
        item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(offset, offset)])
        item.set_identity(identity)
        # Simulate a collision across all model-visible IDs, including the
        # particularly important case of a one-token image span.
        item.pad_value = 1 << 40
        item.pad_values = (item.pad_value,)
        tokens[offset] = item.pad_value
        items.append(item)
    req = Req(
        rid="test",
        origin_input_text=None,
        origin_input_ids=tokens,
        sampling_params=SamplingParams(),
        cache_salt=salt,
    )
    if items:
        req.extend_image_inputs(
            MultimodalInputs(mm_items=items), cache_media_by_position=True
        )
    return req


class TestMultimodalRadixKeys(unittest.TestCase):
    def test_append_and_replace_on_the_python_tree(self):
        page_size = 128
        first_image = page_size + 5
        tokens = [17] * (4 * page_size)
        image_a, image_b = b"a" * 32, b"b" * 32
        first = _request(tokens, [(first_image, image_a)])
        appended = _request(
            tokens + [19] * (2 * page_size),
            [(first_image, image_a), (len(tokens) + 7, image_b)],
        )
        replaced = _request(tokens, [(first_image, image_b)])
        other_tenant = _request(tokens, [(first_image, image_a)], salt="other")
        self.assertEqual(first.origin_input_ids, replaced.origin_input_ids)
        self.assertEqual(first.cache_salt, "tenant")
        self.assertEqual(appended.cache_salt, "tenant")

        cache = SimpleNamespace(
            enable_session_radix_cache=False, window_reprefill_tail_tokens=lambda _: 0
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=page_size,
            tree_slots=tree_slots_of((ComponentType.FULL,)),
        )
        tree = UnifiedTreeCore(
            params,
            {
                ComponentType.FULL: FullComponent(
                    cache, params, slot_of(params, ComponentType.FULL)
                )
            },
        )

        def insert(req):
            key = RadixKey.from_req(req, req.origin_input_ids)
            step = tree.begin_insert(
                InsertParams(key=key, value=torch.arange(len(key), dtype=torch.int64))
            )
            while step.result is None:
                step = tree.resume_insert()
            tree.end_insert()

        def hit(req):
            key = _req_radix_key(cache, req, req.origin_input_ids)
            return tree.match_prefix(MatchPrefixParams(key=key)).device_prefix_len

        insert(first)
        self.assertEqual(hit(appended), len(tokens))
        self.assertEqual(hit(replaced), page_size)
        self.assertEqual(hit(other_tenant), 0)
        self.assertEqual(hit(_request(tokens, [])), page_size)
        insert(appended)
        self.assertEqual(hit(appended), len(appended.origin_input_ids))
        self.assertEqual(hit(first), len(tokens))
        self.assertEqual(hit(replaced), page_size)

    def test_bigram_boundary_uses_the_identity_in_both_slices(self):
        tokens = array("q", [1] * 33)
        left = RadixKey(tokens, is_bigram=True, media_spans=((16, 1, b"a" * 32),))
        right = RadixKey(tokens, is_bigram=True, media_spans=((16, 1, b"b" * 32),))

        self.assertEqual(left.match(right), 15)
        self.assertEqual(left.match(right, page_size=16), 0)
        self.assertNotEqual(left[:16].child_key(16), right[:16].child_key(16))
        self.assertNotEqual(left[16:].child_key(16), right[16:].child_key(16))
        self.assertEqual(left[16:].match_at(left, offset=16), 16)

    def test_missing_offset_retains_the_request_namespace(self):
        req = _request([1] * 8, [(4, b"a" * 32)])
        req.multimodal_inputs.mm_items[0].offsets = None
        req.bind_multimodal_cache_namespace(cache_media_by_position=True)

        self.assertFalse(req.media_cache_spans)
        self.assertTrue(req.cache_salt.startswith("sglang-mm-v1:"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "native KV hash needs Linux")
    def test_hash_changes_at_the_media_page_and_chains_after_it(self):
        page_size = 128
        tokens = [1] * (3 * page_size)
        offset = page_size + 6
        left = _request(tokens, [(offset, b"a" * 32)])
        right = _request(tokens, [(offset, b"b" * 32)])
        keys = [RadixKey.from_req(req, req.origin_input_ids) for req in (left, right)]
        for hash_fn in (get_hash_str, get_storage_hash_str):
            with self.subTest(hash_fn=hash_fn.__name__):
                a, b = [hash_fn(key, page_size=page_size) for key in keys]
                self.assertEqual(a[0], b[0])
                self.assertNotEqual(a[1], b[1])
                self.assertNotEqual(a[2], b[2])
                split_tail = hash_fn(keys[0][page_size:], a[0], page_size=page_size)
                self.assertEqual(a[1:], split_tail)
