"""Deterministic generation, optional parallel and prefix-replay HTTP checks."""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def generate(url, text, max_tokens=32, *, routed_dp_rank=None):
    payload = {
        "text": text,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31088")
    parser.add_argument("--parallel-requests", type=int, default=0)
    parser.add_argument("--prefix-replay", action="store_true")
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
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
