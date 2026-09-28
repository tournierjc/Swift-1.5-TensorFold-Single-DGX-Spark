"""Measure serving speed from the client side: time to first token, prefill rate, decode rate.

Three workloads, because they stress different parts of the engine:

  prose    a short prompt asking for a long French text        -> decode-bound
  code     a short prompt asking for Python with tests         -> decode-bound, code distribution
  prefill  a long pasted context plus a one-line question      -> prefill-bound (TTFT at scale)

Each workload runs twice: the first round pays for whatever the prefix cache does not hold yet, the second
shows the warm number. Every round does one streaming request (for TTFT and the inter-token rhythm) and one
non-streaming request (for the server's own `usage`, which is where the token counts below come from).

    python3 bench/speed.py --base http://127.0.0.1:8083 --model swift-1.5
    python3 bench/speed.py --tokens 256 --rounds 3 --json bench/last.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PARAGRAPH = (
    "Une table n-gram associe les suites de jetons d'un contexte à une ligne de vecteurs appris ; le modèle "
    "la consulte à chaque pas pour rappeler ce que le contexte récent suggère, puis la projette avec le reste "
    "de la couche. La table est énorme et creuse : on ne la garde pas en mémoire, on la lit par petits blocs "
    "au fil des requêtes, ce qui met la latence des accès disque sur le chemin critique du décodage. "
)
PROSE = "Écris un essai de quatre cents mots sur la lecture lente et ce qu'elle change à une époque pressée."
CODE = (
    "Écris une fonction Python `merge_intervals(intervals)` qui fusionne des intervalles fermés qui se "
    "chevauchent, avec sa docstring, ses cas limites et des tests pytest. Réponds uniquement par le code."
)
SUMMARY = "Résume ce texte en une phrase."


def post(base: str, payload: dict):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=7200)


def stream_round(base: str, model: str, messages: list[dict], tokens: int) -> dict:
    """One streaming request: TTFT, total wall time, delta count and the server's usage when it sends one."""

    payload = {"model": model, "messages": messages, "max_tokens": tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    first, chunks, usage, text = None, 0, None, []
    try:
        response = post(base, payload)
    except urllib.error.HTTPError as exc:                      # no stream_options: ask again without it
        if exc.code != 400:
            raise
        payload.pop("stream_options")
        start = time.perf_counter()
        response = post(base, payload)
    with response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter() - start
                    chunks += 1
                    text.append(piece)
    total = time.perf_counter() - start
    return {"ttft": first, "total": total, "chunks": chunks, "usage": usage, "chars": sum(map(len, text))}


def full_round(base: str, model: str, messages: list[dict], tokens: int) -> dict:
    """One non-streaming request: the server's own token accounting and the wall time it took."""

    payload = {"model": model, "messages": messages, "max_tokens": tokens, "temperature": 0, "stream": False}
    start = time.perf_counter()
    with post(base, payload) as response:
        body = json.loads(response.read().decode())
    return {"total": time.perf_counter() - start, "usage": body.get("usage") or {},
            "finish": (body.get("choices") or [{}])[0].get("finish_reason")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure TTFT, prefill and decode speed from the client side.")
    parser.add_argument("--base", default="http://127.0.0.1:8083")
    parser.add_argument("--model", default="swift-1.5")
    parser.add_argument("--tokens", type=int, default=512, help="reply tokens per prose/code round")
    parser.add_argument("--context", type=int, default=2400, help="approximate prompt tokens for the prefill case")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--json", default="", help="also write the measurements here")
    args = parser.parse_args(argv)
    base = args.base.rstrip("/")

    repeats = max(1, args.context * 4 // len(PARAGRAPH))                  # ~4 characters a token, roughly
    workloads = [
        ("prose", [{"role": "user", "content": PROSE}], args.tokens),
        ("code", [{"role": "user", "content": CODE}], args.tokens),
        ("prefill", [{"role": "user", "content": PARAGRAPH * repeats + "\n\n" + SUMMARY}], 64),
    ]

    results = []
    for name, messages, tokens in workloads:
        rounds = []
        for i in range(max(1, args.rounds)):
            run = stream_round(base, args.model, messages, tokens)
            full = full_round(base, args.model, messages, tokens)
            rounds.append({"run": run, "full": full})
            print(f"[{name}] round {i + 1}: ttft={run['ttft'] and round(run['ttft'], 3)}s "
                  f"stream={round(run['total'], 2)}s deltas={run['chunks']} "
                  f"usage={full['usage']} finish={full['finish']}", flush=True)
        warm = rounds[-1]
        prompt = (warm["full"]["usage"] or {}).get("prompt_tokens")
        completion = (warm["run"]["usage"] or warm["full"]["usage"] or {}).get("completion_tokens")
        ttft = warm["run"]["ttft"]
        row = {
            "workload": name,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "ttft_s": round(ttft, 3) if ttft else None,
            "stream_total_s": round(warm["run"]["total"], 2),
            "deltas": warm["run"]["chunks"],
            "nonstream_total_s": round(warm["full"]["total"], 2),
            "prefill_tokens_s": round(prompt / ttft) if prompt and ttft else None,
            "decode_tokens_s": round((completion - 1) / (warm["run"]["total"] - ttft), 1) if completion and ttft else None,
            "rounds": [{"ttft_s": r["run"]["ttft"], "stream_total_s": round(r["run"]["total"], 2),
                        "nonstream_total_s": round(r["full"]["total"], 2),
                        "usage": r["full"]["usage"]} for r in rounds],
        }
        results.append(row)

    print("\n| workload | prompt tok | reply tok | TTFT s | decode tok/s | prefill tok/s | stream s | non-stream s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(f"| {r['workload']} | {r['prompt_tokens']} | {r['completion_tokens']} | {r['ttft_s']} | "
              f"{r['decode_tokens_s']} | {r['prefill_tokens_s']} | {r['stream_total_s']} | {r['nonstream_total_s']} |")
    print("\ndecode tok/s = (completion_tokens - 1) / (stream total - TTFT); prefill = prompt_tokens / TTFT "
          "(the first token's own decode is inside TTFT). Tokens come from the server's usage.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"base": base, "model": args.model, "tokens": args.tokens, "context": args.context,
                       "rounds": args.rounds, "results": results}, fh, indent=2)
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
