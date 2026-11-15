"""V4.1 n-gram IDs/history and original-basis Engram gate, all on device.

Projection remains an explicit BF16 GEMM boundary. Imports never build kernels.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    load_library,
    record_inputs,
)


class EngramOps:
    """Caller-owned tensors/streams; retain them for the whole captured Graph."""

    def __init__(self, *, library=None, device="npu:0"):
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("requires NPU")
        if self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.library = load_library(native_library_path(library))

    def _launch(self, name, tensors, *integers):
        stream = torch.npu.current_stream(self.device)
        record_inputs(self, tensors, stream)
        getattr(self.library, f"engram_{name}")(
            *tensors,
            *integers,
        )

    def hash(self, ids, positions, cu, slots, history, active, parameters, *, out=None):
        out = (
            ids.new_empty(
                (
                    parameters.multipliers.shape[0],
                    ids.numel(),
                    parameters.offsets.shape[1],
                ),
                dtype=torch.int64,
            )
            if out is None
            else out
        )
        self._launch(
            "hash",
            (
                ids,
                positions,
                cu,
                slots,
                history,
                active,
                parameters.token_map,
                parameters.multipliers,
                parameters.primes,
                parameters.offsets,
                out,
            ),
            parameters.pad_id,
            parameters.image_id,
            parameters.mm_shift,
        )
        return out

    def commit(self, ids, cu, slots, accepted, history):
        self._launch("commit", (ids, cu, slots, accepted, history))
        return history

    def gate(self, hidden, kv, q_weight, k_weight, active, *, out=None):
        out = torch.empty_like(hidden) if out is None else out
        self._launch("gate", (hidden, kv, q_weight, k_weight, active, out))
        return out

    def bind_hash(self, ids, positions, cu, slots, history, active, parameters, *, out):
        self.library.prepare_engram_hash(
            ids,
            positions,
            cu,
            slots,
            history,
            active,
            parameters.token_map,
            parameters.multipliers,
            parameters.primes,
            parameters.offsets,
            out,
            parameters.pad_id,
            parameters.image_id,
            parameters.mm_shift,
        )
        return partial(
            self.hash, ids, positions, cu, slots, history, active, parameters, out=out
        )

    def bind_commit(self, ids, cu, slots, accepted, history):
        self.library.prepare_engram_commit(ids, cu, slots, accepted, history)
        return partial(self.commit, ids, cu, slots, accepted, history)

    def bind_gate(self, hidden, kv, q_weight, k_weight, active, *, out):
        """Prepare actual geometry and retain stable Graph tensors."""
        self.library.prepare_engram_gate(hidden, kv, q_weight, k_weight, active, out)
        return partial(
            self._launch, "gate", (hidden, kv, q_weight, k_weight, active, out)
        )


class HashParameters:
    """Startup-validated CPU parameters, copied once to the requested NPU."""

    def __init__(
        self,
        token_map,
        multipliers,
        primes,
        offsets,
        *,
        pad_id,
        image_id=129264,
        mm_shift=1_000_000,
        device="npu:0",
    ):
        tm = torch.as_tensor(token_map, dtype=torch.int64, device="cpu").contiguous()
        mult = torch.as_tensor(
            multipliers, dtype=torch.int64, device="cpu"
        ).contiguous()
        pr = torch.as_tensor(primes, dtype=torch.int64, device="cpu").contiguous()
        off = torch.as_tensor(offsets, dtype=torch.int64, device="cpu").contiguous()
        if (
            tm.ndim != 1
            or not tm.numel()
            or mult.ndim != 2
            or mult.shape[0] <= 0
            or mult.shape[1] <= 1
            or pr.ndim != 3
            or pr.shape[:2] != (mult.shape[0], mult.shape[1] - 1)
            or pr.shape[2] <= 0
            or off.shape != (mult.shape[0], pr.shape[1] * pr.shape[2])
        ):
            raise ValueError(
                "hash requires map[V], multipliers[L,order], primes[L,order-1,heads], offsets[L,(order-1)*heads]"
            )
        maximum = max(int(tm.max()), int(pad_id))
        if int(tm.min()) < 0 or maximum >= 2**31 or not 0 <= pad_id < 2**31:
            raise ValueError("compressed IDs must be nonnegative INT32")
        if int(mult.min()) <= 0 or int(mult.max()) * maximum > 2**63 - 1:
            raise ValueError("hash products must fit nonnegative INT64")
        if int(pr.min()) <= 0 or int(off.min()) < 0:
            raise ValueError("positive primes/nonnegative offsets required")
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("requires NPU")
        if self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.pad_id, self.image_id, self.mm_shift = (
            int(pad_id),
            int(image_id),
            int(mm_shift),
        )
        self.token_map = tm.int().to(self.device)
        self.multipliers = mult.to(self.device)
        self.primes = pr.to(self.device)
        self.offsets = off.to(self.device)
        # Startup-only synchronization permits later calls on arbitrary streams.
        torch.npu.synchronize(self.device)

    @staticmethod
    def model_arrays(model=None):
        """Exact tokenizer normalization/seed/prime order, no SGLang import.

        CPU startup only. Returns arrays suitable for this constructor and
        verifies both declared table sizes and compressed vocabulary cardinality.
        """
        import json
        import os

        from tokenizers import Regex, Tokenizer, normalizers

        model = Path(model or os.environ["DSV41_MODEL_PATH"])
        config = json.loads((model / "config.json").read_text())
        c = config["text_config"]
        tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), "\ue000"),
                normalizers.Strip(),
                normalizers.Replace("\ue000", " "),
            ]
        )
        keys, lookup = {}, []
        for token in range(tokenizer.get_vocab_size(with_added_tokens=True)):
            text = tokenizer.decode([token], skip_special_tokens=False)
            key = (
                tokenizer.id_to_token(token)
                if "\ufffd" in text
                else (normalizer.normalize_str(text) or text)
            )
            if key not in keys:
                keys[key] = len(keys)
            lookup.append(keys[key])
        if len(keys) != c["engram_compressed_vocab_size"]:
            raise ValueError(
                f"tokenizer compressed size {len(keys)} differs from checkpoint"
            )
        bound = max(1, ((2**63 - 1) // len(keys)) // 2)
        mult = np.stack(
            [
                np.random.default_rng(10007 * layer).integers(
                    0, bound, c["engram_max_ngram_size"], dtype=np.int64
                )
                * 2
                + 1
                for layer in c["engram_layer_ids"]
            ]
        )

        # Trial division is startup-only for 48 small (~16M) primes.
        def is_prime(n):
            if n < 2:
                return False
            if n % 2 == 0:
                return n == 2
            d = 3
            while d * d <= n:
                if n % d == 0:
                    return False
                d += 2
            return True

        seen, prime_rows = set(), []
        for _ in c["engram_layer_ids"]:
            row = []
            for _ in range(c["engram_max_ngram_size"] - 1):
                current = c["engram_vocab_size"] - 1
                for _ in range(c["engram_n_heads"]):
                    while not is_prime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    row.append(current)
            prime_rows.append(row)
        primes = np.array(prime_rows, dtype=np.int64)
        offsets = np.stack([np.cumsum(np.r_[0, row[:-1]]) for row in primes])
        if primes.sum(1).tolist() != c["engram_num_embeddings"]:
            raise ValueError("prime layout differs from checkpoint table sizes")
        return {
            "token_map": lookup,
            "multipliers": mult,
            "primes": primes.reshape(
                len(c["engram_layer_ids"]),
                c["engram_max_ngram_size"] - 1,
                c["engram_n_heads"],
            ),
            "offsets": offsets,
            "pad_id": lookup[c["engram_pad_token_id"]],
            "image_id": config.get("image_token_id", -1),
        }
