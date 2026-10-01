#!/usr/bin/env python3
"""Aggregate decode rate over N concurrent clients against a running serve.

    docker run --rm --network host -v "$PWD/bench:/bench" --entrypoint python3 swift-tensorfold:local \\
      /bench/aggregate.py --base http://127.0.0.1:8083 --model qwen3.8-flash-next --clients 1 2 3

A round is weight-bound, so the lanes share its reads and the aggregate is the number that shows it. Two rules
this rig learned measuring it: the clients must send **distinct** prompts (three clients sharing a prefix have
their lanes fight over one cached prefix, worth ~5 tok/s at three lanes), and the reply budget must be long
enough that the per-round setup does not dominate (1024 tokens here). Each client's own rate is printed too, so
an unbalanced run is visible rather than averaged away.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request

# Distinct topics, distinct lengths: nothing here may share a cached prefix with anything else.
TOPICS = [
    "Redige une note technique en francais sur la mise en cache des prefixes de conversation dans un serveur "
    "d'inference, en detaillant le decoupage en blocs, l'eviction et ce qui se mesure.",
    "Write a technical note in English about NVFP4 block quantization: the E2M1 grid, the per-block E4M3 "
    "scales, the global weight_scale_2 factor and what the dequantization costs per matmul.",
    "Escribe una nota tecnica en espanol sobre el muestreo de fotogramas de un video antes de enviarlo a un "
    "modelo de vision: la tasa, el limite, y como se agrupan los fotogramas en bloques temporales.",
]


def one(base: str, model: str, index: int, tokens: int) -> tuple[int, int, float]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": TOPICS[index % len(TOPICS)]}],
                       "max_tokens": tokens}).encode()
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=1800) as reply:
        payload = json.load(reply)
    return payload["usage"]["completion_tokens"], payload["usage"]["prompt_tokens"], time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate decode rate over N concurrent clients.")
    parser.add_argument("--base", default="http://127.0.0.1:8083")
    parser.add_argument("--model", default="swift-1.5")
    parser.add_argument("--tokens", type=int, default=1024, help="reply tokens per client")
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 2, 3], help="client counts to measure")
    args = parser.parse_args()

    for clients in args.clients:
        with concurrent.futures.ThreadPoolExecutor(clients) as pool:
            start = time.time()
            results = list(pool.map(lambda i: one(args.base, args.model, i, args.tokens), range(clients)))
        span = time.time() - start
        done = sum(count for count, _, _ in results)
        each = ", ".join(f"{count / took:.1f}" for count, _, took in results)
        print(f"{clients} client(s): {done} tokens in {span:.2f} s = {done / span:.1f} tok/s aggregate "
              f"(per client {each}; prompts {[prompt for _, prompt, _ in results]} tokens)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
