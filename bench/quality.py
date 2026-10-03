"""Score a frozen item set through /v1/decisions: nothing is generated, so two arms are comparable.

Why this benchmark. Speed is not quality, and a chat conversation cannot measure quality: it samples,
it drifts, and its answer depends on how much of it the prefix cache held. ``/v1/decisions`` scores one
prompt-lane prefill and returns the *logits of the answer labels* at the decision row - no sampling, no
generation, thinking forced off. Those logits are an exact function of the prompt tokens, so the same
item set can be replayed on every arm and the differences that remain are the engine's fidelity, not the
harness's.

What it records per item, and how two runs are compared:

- ``probabilities``  the softmax over the label logits (temperature 1: no reshaping of the decision)
- ``label_mass``     the label set's mass in the *full-vocabulary* distribution, so it is independent of
                     the temperature and of the option count: the share of belief the model puts on any
                     of the offered answers. A wrong-but-confident arm keeps its mass, an arm that has
                     lost the decision drops it.
- ``choice``/``score`` the argmax or the expected level: the human-readable answer.

From ``probabilities`` and ``label_mass`` the absolute log-probability of every label token follows
(log p - log label_mass), so comparing a run against a reference gives the per-token |dlogp| that
published quant tables quote, a Jensen-Shannon divergence between the two decisions, and the flip rate.

Freshness. The prefix cache and the kept conversation snapshots change *timing*, never logits, so a
replayed item scores identically - which is why this harness can be trusted where a chat-based one
cannot. What must be fresh is the run: one file per run, never overwritten, with the prompt token ids
recorded, so two runs that disagree about the tokens (a different arm, a changed chat template, a
different tokenizer) are refused as incomparable instead of being averaged into a number nobody can
interpret. ``scripts/quality-suite.sh`` is the wrapper that enforces this and captures the served arm's
own startup line beside the file.

    python3 bench/quality.py --base http://127.0.0.1:8083 --arm nvfp4-ukisai \\
        --json bench/quality/nvfp4-ukisai.json
    python3 bench/quality.py --compare bench/quality/nvfp4-ukisai.json \\
        --json bench/quality/exl3-405.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request

DEFAULT_ITEMS = "bench/quality-items.json"


def post(base: str, path: str, payload: dict, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"[quality] {path} refused the request ({exc.code}): {detail}") from exc


def canonical(value: object) -> str:
    """The part of a response that must be bit-identical when the same item is replayed."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def answer_name(answer: dict) -> str:
    """The item's answer as a label name: the choice, the yes/no argmax, or the argmax level."""

    if answer["type"] == "choice":
        return answer["choice"]
    probabilities = answer["probabilities"]
    return max(probabilities, key=lambda name: probabilities[name])


def score_item(base: str, model: str, item: dict, timeout: int) -> dict:
    """One question, one request: a failure names the item instead of poisoning a batch."""

    question = dict(item["question"])
    body = {
        "model": model,
        "input": item["input"],
        "questions": [question],
        "temperature": 1.0,                 # 1: the label softmax is reported unreshaped
        "return_prompt_token_ids": True,    # the comparability check between runs
    }
    response = post(base, "/v1/decisions", body, timeout)
    answer = response["answers"][question["id"]]
    return {
        "id": item["id"],
        "category": item.get("category"),
        "gold": item.get("gold"),
        "type": answer["type"],
        "probabilities": answer["probabilities"],
        "label_mass": answer["label_mass"],
        "choice": answer.get("choice"),
        "score": answer.get("score"),
        "label_token_ids": answer["label_token_ids"],
        "prompt_token_ids": answer["prompt_token_ids"],
        "prompt_tokens": len(answer["prompt_token_ids"]),
        "correct": (None if item.get("gold") is None else answer_name(answer) == item["gold"]),
        "sha256": hashlib.sha256(canonical(answer).encode()).hexdigest(),
    }


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (natural log, nats) between two decisions over the same label names."""

    total = 0.0
    for name in p:
        a, b = p[name], q[name]
        middle = (a + b) / 2
        if a > 0:
            total += 0.5 * a * math.log(a / middle)
        if b > 0:
            total += 0.5 * b * math.log(b / middle)
    return total


def logprob(row: dict, name: str) -> float:
    """A label's absolute log-probability in the full-vocabulary distribution."""

    return math.log(max(row["probabilities"][name], 1e-300)) - math.log(max(row["label_mass"], 1e-300))


def load_run(path: str) -> dict:
    with open(path) as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("items"), dict):
        raise SystemExit(f"[quality] {path} is not a quality run: no items map (--compare wants a file this "
                         "harness wrote, not its .models.json or .startup.txt sibling)")
    return loaded


def check_comparable(reference: dict, current: dict) -> None:
    """Refuse to compare runs that were not fed the same tokens, or that were not deterministic."""

    for key in ("items_sha256",):
        if reference.get(key) != current.get(key):
            raise SystemExit(f"[quality] not comparable: {key} differs "
                             f"({reference.get(key)} != {current.get(key)})")
    for row in current["items"].values():
        other = reference["items"].get(row["id"])
        if other is None:
            raise SystemExit(f"[quality] not comparable: {row['id']} is absent from the reference run")
        if other["prompt_token_ids"] != row["prompt_token_ids"]:
            raise SystemExit(
                f"[quality] not comparable: {row['id']} tokenizes differently "
                f"({len(other['prompt_token_ids'])} vs {len(row['prompt_token_ids'])} tokens) - "
                "a different chat template, tokenizer or arm")


def compare(reference: dict, current: dict) -> dict:
    """Per-item divergence and the summaries the arm report quotes."""

    rows = []
    for row in current["items"].values():
        other = reference["items"][row["id"]]
        names = list(other["probabilities"])
        biggest = max(names, key=lambda name: other["probabilities"][name])
        rows.append({
            "id": row["id"],
            "category": row.get("category"),
            "jsd": jsd(other["probabilities"], row["probabilities"]),
            "dlogp": abs(logprob(other, biggest) - logprob(row, biggest)),
            "d_label_mass": abs(other["label_mass"] - row["label_mass"]),
            "flip": (other["choice"] != row["choice"]) if other["type"] == "choice" else None,
            "reference_choice": max(other["probabilities"], key=lambda name: other["probabilities"][name]),
            "answer": max(row["probabilities"], key=lambda name: row["probabilities"][name]),
        })
    flips = [row["flip"] for row in rows if row["flip"] is not None]
    summary = {
        "items": len(rows),
        "mean_jsd": sum(row["jsd"] for row in rows) / len(rows),
        "max_jsd": max(row["jsd"] for row in rows),
        "mean_dlogp": sum(row["dlogp"] for row in rows) / len(rows),
        "max_dlogp": max(row["dlogp"] for row in rows),
        "mean_abs_d_label_mass": sum(row["d_label_mass"] for row in rows) / len(rows),
        "max_abs_d_label_mass": max(row["d_label_mass"] for row in rows),
        "flips": sum(flips),
        "choices": len(flips),
    }
    return {"rows": rows, "summary": summary}


def scored_projection(items: list[dict]) -> list[dict]:
    """Only what is sent to the server: a harness-side annotation can be corrected without invalidating a run."""

    return [{"id": item["id"], "input": item["input"], "question": item["question"]} for item in items]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a frozen item set through /v1/decisions.")
    parser.add_argument("--base", default="http://127.0.0.1:8083")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--items", default=DEFAULT_ITEMS)
    parser.add_argument("--arm", default="", help="the arm name this run measures, recorded in the file")
    parser.add_argument("--json", default="", help="where to write this run (never overwritten)")
    parser.add_argument("--compare", default="", help="a previous run to compare against")
    parser.add_argument("--repeat", type=int, default=1,
                        help="score the whole set this many times and fail on a differing sha256")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--force", action="store_true", help="overwrite --json if it exists")
    args = parser.parse_args(argv)
    base = args.base.rstrip("/")

    with open(args.items) as stream:
        frozen = json.load(stream)
    items = frozen["items"]
    items_sha = hashlib.sha256(canonical(scored_projection(items)).encode()).hexdigest()
    print(f"[quality] {len(items)} items from {args.items} (sha256 {items_sha[:16]} of what is scored) on {base}")

    if args.json:
        try:
            with open(args.json) as stream:
                existing = json.load(stream)
        except FileNotFoundError:
            existing = None
        if existing is not None and not args.force:
            raise SystemExit(f"[quality] {args.json} exists (arm {existing.get('arm')!r}, "
                             f"{existing.get('started')}): a run is never reused, pick another name or --force")

    rows: dict[str, dict] = {}
    for round_index in range(max(1, args.repeat)):
        for item in items:
            row = score_item(base, args.model, item, args.timeout)
            previous = rows.get(row["id"])
            if previous is not None and previous["sha256"] != row["sha256"]:
                raise SystemExit(f"[quality] {row['id']} is not deterministic: "
                                 f"{previous['sha256'][:16]} != {row['sha256'][:16]} "
                                 "(the same tokens must score the same)")
            rows[row["id"]] = row
        print(f"[quality] round {round_index + 1}/{args.repeat} scored, "
              f"{len(rows)} items, all reproducible" if round_index else
              f"[quality] round 1 scored, {len(rows)} items", flush=True)

    current = {
        "arm": args.arm,
        "base": base,
        "model": args.model,
        "items_file": args.items,
        "items_sha256": items_sha,
        "items": rows,
    }

    print("\n| item | kind | prompt tok | answer | label_mass | top |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows.values():
        top = max(row["probabilities"], key=lambda name: row["probabilities"][name])
        answer = row["choice"] if row["type"] == "choice" else (
            f"level {top} (expected {row['score']})" if row["score"] is not None else top)
        print(f"| {row['id']} | {row['type']} | {row['prompt_tokens']} | {answer} | "
              f"{row['label_mass']:.4f} | {top} {row['probabilities'][top]:.4f} |")
    graded = [row for row in rows.values() if row["correct"] is not None]
    if graded:
        right = sum(1 for row in graded if row["correct"])
        print(f"\naccuracy {right}/{len(graded)} = {right / len(graded):.3f} on the {len(graded)} items "
              "with a known answer (a sanity check on the arm, not a quality metric)")

    if args.compare:
        reference = load_run(args.compare)
        check_comparable(reference, current)
        result = compare(reference, current)
        print(f"\n| item | jsd | dlogp | d label_mass | flip | {reference.get('arm') or 'reference'} -> "
              f"{args.arm or 'this run'} |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in result["rows"]:
            print(f"| {row['id']} | {row['jsd']:.4f} | {row['dlogp']:.4f} | {row['d_label_mass']:.4f} | "
                  f"{'flip' if row['flip'] else ('same' if row['flip'] is not None else '-')} | "
                  f"{row['reference_choice']} -> {row['answer']} |")
        summary = result["summary"]
        print(f"\nagainst {args.compare}: mean jsd {summary['mean_jsd']:.4f} (max {summary['max_jsd']:.4f}), "
              f"mean |dlogp| {summary['mean_dlogp']:.4f} (max {summary['max_dlogp']:.4f}), "
              f"mean |d label_mass| {summary['mean_abs_d_label_mass']:.4f}, "
              f"{summary['flips']}/{summary['choices']} choices flipped")
        print("jsd and dlogp are in nats; dlogp is the quantity published quant tables quote, and it is "
              "architectural in this engine: the same weights in a different kernel move it.")
        current["comparison"] = {"reference": args.compare, **summary, "rows": result["rows"]}

    if args.json:
        with open(args.json, "w") as stream:
            json.dump(current, stream, indent=2, sort_keys=True)
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
