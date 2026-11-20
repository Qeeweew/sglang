"""Export target V4.1 MoEs without calibration; original files are read-only."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np


FORMAT = "quarot_h32_int4_g32_v1"
EXPERT = re.compile(
    r"^layers\.\d+\.ffn\.(experts\.\d+|shared_experts)\.w([123])\.(scale|weight)$"
)
FOLDED = re.compile(r"layers\.\d+\.(ffn_norm|ffn\.gate)\.weight")
SIGNS = np.array([1 if 0x9E3779B9 >> bit & 1 else -1 for bit in range(32)], np.float32)


def header(path):
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(length)), length + 8


def shared(raw, exponent, projection, fold):
    code = np.arange(256)
    mag = code % 128
    value = np.ldexp(
        np.where(mag < 8, mag / 8, 1 + (mag % 8) / 8), np.maximum(mag // 8, 1) - 7
    )
    if (mag[raw] == 127).any() or (exponent == 255).any():
        raise ValueError("non-finite shared expert")
    table = (value * np.where(code < 128, 1, -1)).astype(np.float32)
    rows, depth = raw.shape
    weight = table[raw].reshape(rows // 32, 32, depth // 32, 32)
    weight *= np.exp2(exponent.astype(np.float32) - 127)[:, None, :, None]
    weight *= SIGNS
    for width in (1, 2, 4, 8, 16):
        pair = weight.reshape(-1, 2, width)
        left = pair[:, 0].copy()
        np.add(left, pair[:, 1], out=pair[:, 0])
        np.subtract(left, pair[:, 1], out=pair[:, 1])
    weight = weight.reshape(rows, depth) / np.float32(math.sqrt(32))
    if projection == 3:
        weight *= np.tile(SIGNS, rows // 32)[:, None]
    if projection != 2:
        weight = weight.reshape(-1, 8, 32, depth).swapaxes(1, 2).reshape(rows, depth)
    unit = np.max(np.abs(weight), axis=1, keepdims=True) / 127
    unit[unit == 0] = 1
    return np.rint(weight / unit).clip(-127, 127).astype(np.int8), unit * np.float32(
        fold
    )


def prepare_worker(threads):
    cpus = sorted(os.sched_getaffinity(0))
    slot = mp.current_process()._identity[-1] - 1
    start = slot * threads % len(cpus)
    os.sched_setaffinity(0, (cpus * 2)[start : start + threads])


@cache
def routed_fitter(library):
    convert = ctypes.CDLL(str(library)).export_g32
    convert.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int]
    )
    return convert


def shard(args):
    src, dst, lib, threads = args
    source, start = header(src)
    meta = source.pop("__metadata__", None)
    keys = sorted(source, key=lambda name: source[name]["data_offsets"][0])
    if not any(EXPERT.fullmatch(key) or FOLDED.fullmatch(key) for key in keys):
        if not dst.exists():
            os.link(src, dst)
        if not os.path.samefile(src, dst):
            raise ValueError(f"not original shard: {dst}")
        return src.stat().st_size - start
    plan = {}
    cursor = 0
    for key in keys:
        record = source[key]
        dtype, shape = record["dtype"], record["shape"]
        first, last = record["data_offsets"]
        size = last - first
        match = EXPERT.fullmatch(key)
        if match:
            kind, _, field = match.groups()
            if field == "scale":
                rows, depth = source[key[:-5] + "weight"]["shape"]
                expected = (
                    [rows // 32, depth // 32]
                    if kind == "shared_experts"
                    else [rows, depth // 16]
                )
                if dtype != "F8_E8M0" or shape != expected:
                    raise ValueError(
                        f"unsupported source scale: {key}: {dtype} {shape}"
                    )
                if kind == "shared_experts":
                    shape, size, dtype = [rows, 1], 4 * rows, "F32"
                else:
                    shape, dtype = [rows, depth // 32 + 4], "U8"
                    size = math.prod(shape)
            else:
                if dtype != ("F8_E4M3" if kind == "shared_experts" else "I8"):
                    raise ValueError(f"unsupported source weights: {key}: {dtype}")
                dtype = "I8" if kind == "shared_experts" else "U8"
                if shape[0] % 256 or shape[1] % 128 or size != math.prod(shape):
                    raise ValueError(f"unsupported expert dimensions: {key}: {shape}")
        plan[key] = dict(dtype=dtype, shape=shape, data_offsets=[cursor, cursor + size])
        cursor += size
    if dst.exists():
        old, base = header(dst)
        old.pop("__metadata__", None)
        if old != plan or dst.stat().st_size != base + cursor:
            raise ValueError(f"incompatible completed shard: {dst}")
        return cursor
    packed_header = json.dumps(
        dict(plan, **({"__metadata__": meta} if meta else {})), separators=(",", ":")
    ).encode()
    packed_header += b" " * (-len(packed_header) % 8)
    temp = dst.with_suffix(".partial")
    with temp.open("wb") as stream:
        stream.write(struct.pack("<Q", len(packed_header)))
        stream.write(packed_header)
        stream.truncate(stream.tell() + cursor)
    origin = np.memmap(src, mode="r", offset=start, dtype="u1")
    target = np.memmap(temp, mode="r+", offset=len(packed_header) + 8, dtype="u1")

    def view(blob, spec, name, dtype="u1"):
        first, last = spec[name]["data_offsets"]
        return blob[first:last].view(dtype).reshape(spec[name]["shape"])

    for key in keys:
        match = EXPERT.fullmatch(key)
        if not match:
            first, last = source[key]["data_offsets"]
            left, right = plan[key]["data_offsets"]
            folded = FOLDED.fullmatch(key)
            if folded:
                if source[key]["dtype"] != "BF16" or source[key]["shape"][-1] % 32:
                    raise ValueError(f"expected BF16 with complete H32 groups: {key}")
                inp = view(origin, source, key, "<u2").astype(np.uint32) << 16
                vals = inp.view(np.float32).reshape(-1, 32) * SIGNS
                vals *= 0.25 if folded[1] == "ffn_norm" else 4
                out = view(target, plan, key, "<u2")
                out[:] = (vals.view(np.uint32) >> 16).reshape(out.shape)
            else:
                target[left:right] = origin[first:last]
            continue
        kind, proj, field = match.groups()
        if field == "scale":
            continue
        proj = int(proj)
        scale = key[:-6] + "scale"
        raw, exps = view(origin, source, key), view(origin, source, scale)
        output = view(target, plan, key)
        fold = (1 if proj == 2 else 4) / math.sqrt(32)
        if kind == "shared_experts":
            out, units = shared(raw, exps, proj, fold)
            output[:] = out.view(np.uint8)
            view(target, plan, scale, "<f4")[:] = units
        else:
            units = view(target, plan, scale)
            rows, packed = raw.shape
            if routed_fitter(lib)(
                *(arr.ctypes.data for arr in (raw, exps, output, units)),
                rows,
                packed * 2,
                threads,
                fold,
                proj,
            ):
                raise ValueError(f"invalid source {key}")
    target.flush()
    del target
    os.replace(temp, dst)
    print(f"finished {dst.name} ({cursor} tensor bytes)", flush=True)
    return cursor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "library"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=20)
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    if (
        src == dst
        or src in dst.parents
        or dst in src.parents
        or not args.library.is_file()
    ):
        raise ValueError(
            "expected a separate destination and a compiled export_g32 library"
        )
    if min(args.threads, args.workers) <= 0:
        raise ValueError("worker and thread counts must be positive")
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / ".g32-export.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        export(args, src, dst)


def export(args, src, dst):
    idx = "model.safetensors.index.json"
    index = json.loads((src / idx).read_text())
    files = sorted(set(index["weight_map"].values()))
    if (dst / "config.json").exists():
        raise FileExistsError(f"refusing to overwrite a published checkpoint: {dst}")
    stamp = dst / ".g32-source.json"
    manifest = {
        "format": FORMAT,
        "source": str(src),
        "index": hashlib.sha256((src / idx).read_bytes()).hexdigest(),
        "exporter": hashlib.sha256(
            Path(__file__).read_bytes() + args.library.read_bytes()
        ).hexdigest(),
        "shards": {name: (src / name).stat().st_mtime_ns for name in files},
    }
    if stamp.exists():
        if json.loads(stamp.read_text()) != manifest:
            raise ValueError("partial export belongs to a different source or fitter")
    elif any(path.name != ".g32-export.lock" for path in dst.iterdir()):
        raise FileExistsError(
            f"refusing to reuse an unrelated non-empty directory: {dst}"
        )
    else:
        stamp.write_text(json.dumps(manifest))
    with ProcessPoolExecutor(
        args.workers, initializer=prepare_worker, initargs=(args.threads,)
    ) as pool:
        total = sum(
            pool.map(
                shard,
                [
                    (src / name, dst / name, args.library.resolve(), args.threads)
                    for name in files
                ],
            )
        )
    for path in src.iterdir():
        if path.name in (*files, idx, "config.json"):
            continue
        if path.is_dir():
            shutil.copytree(path, dst / path.name, dirs_exist_ok=True)
        else:
            shutil.copy2(path, dst / path.name)
    config = json.loads((src / "config.json").read_text())
    config["text_config"]["dsv41_target_moe_format"] = FORMAT
    index.setdefault("metadata", {})["total_size"] = total
    for name, obj in ((idx, index), ("config.json", config)):
        temp = dst / (name + ".partial")
        temp.write_text(json.dumps(obj, indent=2) + "\n")
        temp.replace(dst / name)
    print(f"published {len(files)} shards, {total} tensor bytes", flush=True)


if __name__ == "__main__":
    main()
