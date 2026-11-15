"""Deterministic generation, optional parallel and prefix-replay HTTP checks."""

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def generate(url, prompt, max_tokens=32, *, routed_dp_rank=None):
    input_field = "text" if isinstance(prompt, str) else "input_ids"
    payload = {
        input_field: prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        "return_logprob": True,
    }
    if routed_dp_rank is not None:
        payload["routed_dp_rank"] = routed_dp_rank
    response = requests.post(
        url + "/generate",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    result = response.json()
    assert result["meta_info"]["completion_tokens"] > 0, result
    values = result["meta_info"]["output_token_logprobs"]
    assert values and all(math.isfinite(value[0]) for value in values), result
    return result


def chat(url, model, text, max_tokens=32, *, routed_dp_rank=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
    }
    if routed_dp_rank is not None:
        payload["routed_dp_rank"] = routed_dp_rank
    response = requests.post(url + "/v1/chat/completions", json=payload, timeout=600)
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    assert choice["message"]["content"].strip(), result
    values = choice["logprobs"]["content"]
    assert values and all(math.isfinite(value["logprob"]) for value in values), result
    return result


def indexed_kv_blocks(router_url, rank):
    response = requests.get(router_url + "/metrics", timeout=5)
    response.raise_for_status()
    for line in response.text.splitlines():
        if line.startswith("sgl_router_kv_tree_blocks{") and (
            f'dp_rank="{rank}",tier="device"' in line
        ):
            return int(line.rsplit(maxsplit=1)[1])
    return 0


def check_router_prefix(backend_url, router_url):
    # 910B uses 128-token cache pages. Warm a fresh prefix on a known rank.
    prefix = random.choices(range(256, 8192), k=1024)
    rank = 1
    before = indexed_kv_blocks(router_url, rank)
    first = generate(backend_url, prefix + [100], 1, routed_dp_rank=rank)
    seed = first["meta_info"]
    assert seed["dp_rank"] == rank and seed["cached_tokens"] == 0, seed

    deadline = time.monotonic() + 10
    while indexed_kv_blocks(router_url, rank) < before + len(prefix) // 128:
        if time.monotonic() >= deadline:
            raise RuntimeError("Router did not receive the warmed KV pages")
        time.sleep(0.1)

    second = generate(router_url, prefix + [101], 1)
    hit = second["meta_info"]
    assert hit["dp_rank"] == seed["dp_rank"] and hit["cached_tokens"] >= len(prefix), hit
    print(f"router prefix reuse: rank {hit['dp_rank']}, {hit['cached_tokens']} cached tokens")
    return [first, second]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31088")
    parser.add_argument("--parallel-requests", type=int, default=0)
    parser.add_argument("--prefix-replay", action="store_true")
    parser.add_argument("--router-url", help="Also check cross-request prefix routing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.parallel_requests < 0:
        parser.error("--parallel-requests must be nonnegative")
    response = requests.get(args.url + "/v1/models", timeout=30)
    response.raise_for_status()
    model = response.json()["data"][0]["id"]
    prompt = "Explain what a mixture of experts model is in one sentence."
    results = {"single": chat(args.url, model, prompt)}
    print(results["single"]["choices"][0]["message"]["content"], flush=True)
    if args.parallel_requests:
        with ThreadPoolExecutor(max_workers=args.parallel_requests) as executor:
            results["parallel"] = list(
                executor.map(
                    lambda i: chat(
                        args.url, model, f"What is {i} plus 2? Answer briefly.", 16
                    ),
                    range(args.parallel_requests),
                )
            )
        print(f"parallel requests: {len(results['parallel'])} passed", flush=True)
    if args.prefix_replay:
        prefix = "The quick brown fox jumps over the lazy dog. " * 128
        text = prefix + "\nSummarize the repeated sentence briefly."
        # Each attention-DP worker owns its prefix tree. Compare on one worker.
        first = chat(args.url, model, text, 32, routed_dp_rank=0)
        second = chat(args.url, model, text, 32, routed_dp_rank=0)
        cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert cached >= 128, second
        results["prefix"] = [first, second]
        print(f"prefix replay: {cached} cached tokens", flush=True)
    results["native"] = generate(args.url, prompt)
    if args.router_url:
        results["router_prefix"] = check_router_prefix(args.url, args.router_url)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
