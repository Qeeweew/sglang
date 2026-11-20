"""Paired target-only scoring for the 910B MoE A/B, using saved token IDs."""

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import requests


def prepare(args):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    bos = json.loads((args.model / "config.json").read_text())["bos_token_id"]
    rng = random.Random(41019)
    gsm = [json.loads(line) for line in args.gsm.read_text().splitlines()]
    samples = {name: [] for name in ("en", "zh", "code", "math")}
    seen = set()

    def add(kind, text):
        if len(samples[kind]) >= args.docs or text in seen:
            return
        ids = tokenizer.encode(text, add_special_tokens=False).ids[: args.window]
        if len(ids) >= 80:
            samples[kind].append([bos, *ids])
            seen.add(text)
            return True
        return False

    questions = rng.sample(range(len(gsm)), len(gsm))
    for idx in questions[args.greedy :]:
        row = gsm[idx]
        add("math", row["question"] + "\n" + row["answer"])
        if len(samples["math"]) == args.docs:
            break
    chats = json.loads(args.sharegpt.read_text())
    rng.shuffle(chats)
    for chat in chats:
        for turn in chat["conversations"]:
            if turn["from"] not in ("gpt", "assistant"):
                continue
            text = turn["value"].strip()
            if not 100 <= len(text) <= 24000:
                continue
            blocks = re.findall(r"```(?:[\w+#.-]+)?\n(.*?)```", text, re.S)
            chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
            if blocks:
                used = add("code", max(blocks, key=len))
            elif chinese > len(text) * 0.3:
                used = add("zh", text)
            elif chinese == 0 and sum(ch.isascii() for ch in text) > 0.97 * len(text):
                used = add("en", text)
            else:
                continue
            if used:
                break
        if all(len(group) == args.docs for group in samples.values()):
            break
    if not all(len(group) == args.docs for group in samples.values()):
        raise ValueError({k: len(v) for k, v in samples.items()})
    corpus = [
        dict(kind=kind, ids=ids)
        for group in zip(*samples.values())
        for kind, ids in zip(samples, group)
    ]
    examples = "Question: Ana has 5 apples and buys 2 more. How many does she have?\nAnswer: 5 + 2 = 7.\n#### 7\n\nQuestion: A box holds 4 rows of 3 eggs. How many eggs is that?\nAnswer: 4 * 3 = 12.\n#### 12\n\n"
    greedy = []
    for idx in questions[: args.greedy]:
        prompt = examples + "Question: " + gsm[idx]["question"] + "\nAnswer:"
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        greedy.append(
            dict(
                idx=idx,
                ids=[bos, *ids],
                answer=gsm[idx]["answer"].split("#### ")[-1].strip().replace(",", ""),
            )
        )
    args.corpus.write_text(json.dumps(dict(docs=corpus, greedy=greedy)))
    print(
        "scored positions",
        {kind: sum(len(d) - 1 for d in docs) for kind, docs in samples.items()},
        flush=True,
    )


def score(args):
    corpus = json.loads(args.corpus.read_text())
    sha = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    session = requests.Session()

    def generate(ids, **options):
        response = session.post(
            args.url + "/generate", json=dict(input_ids=ids, **options), timeout=300
        )
        response.raise_for_status()
        return response.json()

    def measure(doc):
        ids = doc["ids"]
        meta = generate(
            ids,
            sampling_params=dict(temperature=0, max_new_tokens=1),
            return_logprob=True,
            top_logprobs_num=2,
            logprob_start_len=0,
            routed_dp_rank=0,
        )["meta_info"]
        gold, tops = meta["input_token_logprobs"], meta["input_top_logprobs"]
        if (
            len(gold) != len(ids)
            or len(tops) != len(ids)
            or [v[1] for v in gold[1:]] != ids[1:]
        ):
            raise ValueError(
                "teacher-forcing labels/positions differ from frozen corpus"
            )
        nll = [-v[0] for v in gold[1:]]
        margins = [v[0][0] - v[1][0] for v in tops[1:]]
        if not all(math.isfinite(v) for v in nll + margins):
            raise ValueError("non-finite model output")
        return dict(
            kind=doc["kind"], nll=nll, top=[v[0][1] for v in tops[1:]], margin=margins
        )

    with args.output.open("x", buffering=1) as dest:

        def emit(**data):
            dest.write(json.dumps(dict(corpus_sha256=sha, **data)) + "\n")

        for idx, doc in enumerate(corpus["docs"]):
            capture = args.capture_marker if idx < 4 else None
            if capture:
                capture.write_text(doc["kind"])
            try:
                result = measure(doc)
            finally:
                if capture:
                    capture.unlink(missing_ok=True)
            emit(doc=idx, **result)
            if idx == 0:
                repeat = measure(doc)
                emit(control=idx, **repeat)
            if not (idx + 1) % 16:
                print("scored docs", idx + 1, flush=True)
        for start in range(0, len(corpus["greedy"]), 8):
            batch = corpus["greedy"][start : start + 8]
            results = generate(
                [task["ids"] for task in batch],
                sampling_params=dict(
                    temperature=0,
                    max_new_tokens=192,
                    stop=["\n\nQuestion:", "\nQuestion:"],
                ),
                routed_dp_rank=0,
            )
            for task, result in zip(batch, results, strict=True):
                text = result["text"]
                matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
                pred = matches[-1].replace(",", "") if matches else None
                emit(
                    greedy=task["idx"],
                    correct=pred == task["answer"],
                    pred=pred,
                    gold=task["answer"],
                    text=text,
                    stop=result["meta_info"]["finish_reason"],
                )
            print("greedy tasks", start + len(batch), flush=True)


def compare(args):
    datasets = [
        [json.loads(row) for row in path.read_text().splitlines()]
        for path in (args.baseline, args.candidate)
    ]
    if args.teacher_only:
        datasets = [
            [row for row in records if "greedy" not in row] for records in datasets
        ]
    arms = [{r["doc"]: r for r in records if "doc" in r} for records in datasets]
    tasks = [{r["greedy"]: r for r in records if "greedy" in r} for records in datasets]
    corpus = json.loads(args.corpus.read_text())
    expected_tasks = (
        set() if args.teacher_only else {r["idx"] for r in corpus["greedy"]}
    )
    sha = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    if (
        not all(all(row["corpus_sha256"] == sha for row in ds) for ds in datasets)
        or not all(set(a) == set(range(len(corpus["docs"]))) for a in arms)
        or not all(set(t) == expected_tasks for t in tasks)
    ):
        raise ValueError("incomplete or unpaired data")
    for records, arm in zip(datasets, arms, strict=True):
        if (
            len(records) != len(arm) + len(expected_tasks) + 1
            or sum("control" in row for row in records) != 1
        ):
            raise ValueError("duplicate score or missing repeat control")
        for idx, doc in enumerate(corpus["docs"]):
            if arm[idx]["kind"] != doc["kind"] or not all(
                len(arm[idx][field]) == len(doc["ids"]) - 1
                for field in ("nll", "top", "margin")
            ):
                raise ValueError(f"incomplete document {idx}")
    rng = np.random.default_rng(41019)
    result = {}
    for kind in ("all", "en", "zh", "code", "math"):
        ids = [
            idx
            for idx, doc in enumerate(corpus["docs"])
            if kind == "all" or kind == doc["kind"]
        ]
        counts = np.array([len(corpus["docs"][idx]["ids"]) - 1 for idx in ids])
        nll = np.array([[sum(a[idx]["nll"]) for idx in ids] for a in arms])
        delta = nll[1] - nll[0]
        draws = rng.integers(0, len(ids), (10000, len(ids)))

        def flat(arm, key):
            return np.concatenate([arm[idx][key] for idx in ids])

        predictions = [flat(arm, "top") for arm in arms]
        gold = np.concatenate([corpus["docs"][idx]["ids"][1:] for idx in ids])
        flips = predictions[0] != predictions[1]
        confident = flat(arms[0], "margin") >= 1
        result[kind] = dict(
            documents=len(ids),
            tokens=int(counts.sum()),
            ppl=np.exp(nll.sum(1) / counts.sum()).tolist(),
            delta_nll_nats=float(delta.sum() / counts.sum()),
            delta_ppl_pct=float(100 * np.expm1(delta.sum() / counts.sum())),
            ci95_pct=(
                100
                * np.expm1(
                    np.quantile(
                        delta[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]
                    )
                )
            ).tolist(),
            top1_flips=int(flips.sum()),
            top1_correct=[int((top == gold).sum()) for top in predictions],
            high_margin_tokens=int(confident.sum()),
            high_margin_flips=int((flips & confident).sum()),
        )
    result["control"] = []
    for records, arm in zip(datasets, arms, strict=True):
        repeat = next(row for row in records if "control" in row)
        first = arm[repeat["control"]]
        delta = np.subtract(repeat["nll"], first["nll"])
        flips = np.not_equal(repeat["top"], first["top"])
        result["control"].append(
            dict(
                doc=repeat["control"],
                tokens=len(delta),
                max_abs_delta_nll=float(np.abs(delta).max()),
                delta_ppl_pct=float(100 * np.expm1(delta.mean())),
                top1_flips=int(flips.sum()),
                high_margin_flips=int((flips & (np.array(first["margin"]) >= 1)).sum()),
            )
        )
    if args.teacher_only:
        print(json.dumps(result, indent=2))
        return
    result["gsm8k"] = dict(
        n=len(tasks[0]),
        correct=[sum(r["correct"] for r in t.values()) for t in tasks],
        truncated=[
            sum(r["stop"]["type"] == "length" for r in t.values()) for t in tasks
        ],
        lost=sum(
            tasks[0][idx]["correct"] and not r["correct"] for idx, r in tasks[1].items()
        ),
        gained=sum(
            not tasks[0][idx]["correct"] and r["correct"] for idx, r in tasks[1].items()
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    for mode, fn in [("prepare", prepare), ("score", score), ("compare", compare)]:
        sub = subs.add_parser(mode)
        sub.add_argument("--corpus", type=Path, required=True)
        sub.set_defaults(fn=fn)
        if mode == "prepare":
            for flag in ("--model", "--gsm", "--sharegpt"):
                sub.add_argument(flag, type=Path, required=True)
            sub.add_argument("--docs", type=int, default=32)
            sub.add_argument("--window", type=int, default=192)
            sub.add_argument("--greedy", type=int, default=32)
        elif mode == "score":
            sub.add_argument("--url", required=True)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--capture-marker", type=Path)
        else:
            sub.add_argument("--baseline", type=Path, required=True)
            sub.add_argument("--candidate", type=Path, required=True)
            sub.add_argument(
                "--teacher-only",
                action="store_true",
                help="Require all paired documents, but omit greedy generation results.",
            )
    args = parser.parse_args()
    args.fn(args)
