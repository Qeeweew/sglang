"""Indexed, expert-major V4.1 input with at most two experts of staging.

Only routed FP4 inputs use bounded pread buffers. Dense/shared/vision and
registered Engram inputs retain the ordinary safetensors mmap representation.
One expert is prefetched while the current expert is repacked. No whole-shard
prefetch queue or unbounded asynchronous expert submission is used.
"""

from __future__ import annotations

import json
import logging
import math
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import wraps
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .policy import enabled

logger = logging.getLogger(__name__)
_EXPERT = re.compile(
    r"^((?:layers|mtp)\.\d+\.ffn\.experts\.(\d+))\.(w[123])\.(weight|scale)$"
)
_PARTS = tuple(
    f"{projection}.{kind}"
    for projection in ("w1", "w3", "w2")
    for kind in ("weight", "scale")
)


class ExpertCheckpoint:
    def __init__(self, directory, *, draft=False):
        self.directory = Path(directory).resolve()
        self.draft = draft
        self.index = json.loads(
            (self.directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.headers = {}
        self.shards = {}
        self.groups = {}
        self.other = []
        self.input_bytes = 0
        self.peak_expert_bytes = 0
        self.read_seconds = 0.0
        for name in self.index:
            # Filter before opening shards, including the three native DSpark
            # stages. The draft reader never reads any target expert or table.
            if name.startswith("mtp.") != draft:
                continue
            match = _EXPERT.fullmatch(name)
            if match:
                self.groups.setdefault(match[1], set()).add(name[len(match[1]) + 1 :])
            elif ".experts." in name:
                raise ValueError(f"unsupported V4.1 streaming expert tensor: {name}")
            else:
                self.other.append(name)
        for prefix, parts in self.groups.items():
            if parts != set(_PARTS):
                raise ValueError(
                    f"incomplete V4.1 expert {prefix}: missing {set(_PARTS) - parts}"
                )

    def _read_expert_tensor(self, name):
        shard = self.index[name]
        if shard not in self.shards:
            path = (self.directory / shard).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("checkpoint shard escapes model directory")
            self.shards[shard] = path
        path = self.shards[shard]
        if path not in self.headers:
            with path.open("rb") as source:
                size = struct.unpack("<Q", source.read(8))[0]
                if size > min(path.stat().st_size - 8, 100 * 1024 * 1024):
                    raise ValueError(f"invalid safetensors header: {path}")
                self.headers[path] = (8 + size, json.loads(source.read(size)))
        base, header = self.headers[path]
        meta = header[name]
        first, last = meta["data_offsets"]
        if (
            meta["dtype"] not in {"I8", "U8", "F8_E8M0"}
            or not (0 <= first <= last <= path.stat().st_size - base)
            or last - first != math.prod(meta["shape"])
        ):
            raise ValueError(f"invalid byte-packed V4.1 tensor: {name}")
        # Uninitialized staging; final packed output is separately allocated
        # and repacked directly on the NPU.
        value = torch.empty(meta["shape"], dtype=torch.uint8, device="cpu")
        with path.open("rb", buffering=0) as source:
            source.seek(base + first)
            target = memoryview(value.numpy()).cast("B")
            cursor = 0
            while cursor < len(target):
                count = source.readinto(target[cursor:])
                if not count:
                    raise ValueError(f"truncated V4.1 expert tensor: {name}")
                cursor += count
        self.input_bytes += last - first
        return value

    def _read_expert(self, prefix):
        started = time.perf_counter()
        values = [
            (f"{prefix}.{part}", self._read_expert_tensor(f"{prefix}.{part}"))
            for part in _PARTS
        ]
        self.peak_expert_bytes = max(
            self.peak_expert_bytes, sum(value.numel() for _, value in values)
        )
        self.read_seconds += time.perf_counter() - started
        return values

    def __iter__(self):
        from sglang.srt.utils import BAR_FORMAT

        enable_tqdm = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        role = "draft" if self.draft else "target"
        with (
            closing(self._iter_tensors()) as tensors,
            tqdm(
                total=len(self.groups) * len(_PARTS) + len(self.other),
                desc=f"Loading V4.1 {role} weights",
                unit="tensor",
                disable=not enable_tqdm,
                bar_format=BAR_FORMAT,
                position=tqdm._get_free_pos(),
                mininterval=1.0,
            ) as progress,
        ):
            for name, value in tensors:
                yield name, value
                # Advance after the consumer has loaded/repacked this tensor.
                progress.update()

    def _iter_tensors(self):
        from safetensors import safe_open

        started = time.perf_counter()

        # Numerically ordered layers/experts, with all six pieces adjacent even
        # when the index is weights-first or splits scales across shards.
        def key(name):
            return tuple(
                int(part) if part.isdigit() else part for part in name.split(".")
            )

        prefixes = iter(sorted(self.groups, key=key))
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dsv41-input"
        ) as executor:
            prefix = next(prefixes, None)
            pending = (
                None if prefix is None else executor.submit(self._read_expert, prefix)
            )
            try:
                while pending is not None:
                    values = pending.result()
                    prefix = next(prefixes, None)
                    pending = (
                        None
                        if prefix is None
                        else executor.submit(self._read_expert, prefix)
                    )
                    for name, value in values:
                        # Keep upstream expert consumption inline. Only this
                        # reader owns a bounded one-expert lookahead, and native
                        # repack completes before the next expert.
                        value._sglang_runai_streamer_tensor = True
                        yield name, value
                    del value, values
            finally:
                if pending is not None:
                    pending.cancel()
                # Executor shutdown joins the reader on errors/early close.
        # Keep FP8 pairs adjacent for the BF16 converter as well. Engram source
        # mmap tensors are copied into registered storage by the original hook.
        for name in sorted(self.other, key=lambda name: (name.rsplit(".", 1)[0], name)):
            with safe_open(
                self.directory / self.index[name], framework="pt", device="cpu"
            ) as source:
                value = source.get_tensor(name)
                # There are two owned mmap Engram table pairs. They need no
                # borrowed-buffer restriction: restore their parallel host
                # copies and let the upstream loader join before publication.
                # Other tensors stay inline to bound dense conversion inputs.
                if ".engram.embed." not in name:
                    value._sglang_runai_streamer_tensor = True
                yield name, value
                del value
        logger.debug(
            "V4.1 checkpoint input: %d experts, %.2fs (expert reads %.2fs)",
            len(self.groups),
            time.perf_counter() - started,
            self.read_seconds,
        )


def adapt_iterator(original):
    if not enabled():
        return original

    @wraps(original)
    def iterator(self, source, **kwargs):

        from sglang.srt.hardware_backend.npu.dsv41.dense_weights import _is_v41

        from .policy import enabled as stream_experts_enabled

        config = getattr(source.model_config, "hf_config", None)
        if not stream_experts_enabled() or not _is_v41(config):
            return original(self, source, **kwargs)
        resolved = kwargs.get("resolved_source")
        if resolved is None:
            folder, _, safetensors = self._prepare_weights(
                source.model_or_path,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
        else:
            folder, safetensors = (resolved.hf_folder, resolved.use_safetensors)
        if (
            not safetensors
            or source.prefix
            or self.load_config.draft_model_idx is not None
        ):
            raise ValueError(
                "V4.1 streaming requires native indexed safetensors without prefix/MTP remapping"
            )
        draft = "DeepseekV4ForCausalLMDSpark" in getattr(config, "architectures", ())
        reader = ExpertCheckpoint(folder, draft=draft)
        if not self.counter_before_loading_weights:
            self.counter_before_loading_weights = time.perf_counter()
        return iter(reader)

    return iterator
