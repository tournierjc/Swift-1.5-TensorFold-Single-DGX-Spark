# Swift 1.5 on TensorFold — one DGX Spark

A test rig that serves the Swift 1.5 NVFP4 checkpoint on a single DGX Spark (GB10, 128 GB unified memory)
with [TensorFold](https://github.com/ashhart/TensorFold) 0.5.0: FP4 routed experts, BF16 everywhere else, and
the PLE layer's n-gram table in the BF16 layout this revision publishes.

The sibling rig [`Qwen3.8-Flash-Next-Single-DGX-Spark`](https://github.com/tournierjc/Qwen3.8-Flash-Next-Single-DGX-Spark)
serves the same model family with a patched vLLM; this one runs the TensorFold CUDA route end to end on the
Spark. The image carries the vision path: `transformers==5.17.0`, `av`, and a local port of the `qwen4_exp` vision
patch — a port rather than a flag, because the upstream accepts only `qwen3_5` here. It is dormant unless
`--vision` is passed, and it needs at least two lanes. Verified end to end: a solid red square and a solid blue
square answered "Rouge" (2.7 s) and "Bleu" (2.9 s).

## What it runs

| | |
| --- | --- |
| Model | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`, revision `3ff05202` — 186.4 GB over 296,474 tensors |
| Engine | TensorFold **0.5.0** (`ashhart/TensorFold`; the `Dockerfile` pins a ref, override with `TF_REF`), with three local changes carried on top: the multi-row item-16 pair path, the 8-bit projection copies gated by `TENSORFOLD_FACES_FP8`, and an fp32 reduce / `_fp4mm` block change. A local port of the `qwen4_exp` vision patch rides with them |
| Served as | `qwen3.8-flash-next` on `:8083` |
| Endpoint | OpenAI-compatible (`/health`, `/v1/models`, `/v1/chat/completions`, streaming and tool calls) |
| Speed | `scripts/bench.sh` → `bench/speed.py`: TTFT, prefill rate and decode rate |

The checkpoint's own `ple_embedding.ngram_embedding.shard_N.weight` tensors are BF16 `[2500012, 160]` rows with
no per-shard scales — 128 shards, 320,001,536 rows, 29.8 GiB, memory-mapped and gathered a lookup at a time.

The pinned ref and its upstream PR, the base image, the ModelOpt NVFP4 dequant formula, the vision install and
the `reasoning_content`/`content` split are in [docs/engine-status.md](docs/engine-status.md).

## Requirements

- One DGX Spark / GB10 with the NVIDIA container runtime (`docker run --gpus all` works).
- **About 250 GB free disk**: 186 GB for the checkpoint, ~40 GB for the image, plus room for the kernel caches.
  The n-gram table is read from disk at every lookup, so it belongs on the NVMe, not on a network mount.
- Nothing else: no Python environment on the host, no NGC login (NVIDIA's PyTorch image pulls anonymously).

## Quick start

```bash
git clone https://github.com/tournierjc/Swift-1.5-TensorFold-Single-DGX-Spark.git
cd Swift-1.5-TensorFold-Single-DGX-Spark
cp .env.sample .env                  # optional: edit paths, port, serve flags

scripts/build.sh                     # build the image (local only, no registry)
scripts/pull.sh                      # download the checkpoint: 186 GB, resumable
scripts/serve.sh                     # foreground; Ctrl-C stops it
scripts/smoke.sh                     # in another shell: health, /v1/models, one timed completion
scripts/bench.sh                     # in another shell: prose, code and prefill speed
```

The running image is **not** the `Dockerfile`'s default ref, which pins the upstream. Build the branch it
came from, and pass `TF_REPO` as well — without it the build resolves the upstream and a fork branch is
simply not there (`pathspec ... did not match any file(s) known to git`):

```bash
TF_REPO=https://github.com/tournierjc/TensorFold.git TF_REF=feat/vision-qwen4-exp scripts/build.sh
```

Logs go to the terminal that runs `scripts/serve.sh`; `scripts/stop.sh` stops a detached run and an
interrupted download.

## Serving on 128 GB

`scripts/preflight.py` prints the whole plan from the checkpoint's headers alone, in seconds, before any load.
The current serve reports one startup line, and the whole load took **455.1 s**:

    startup estimate 97.39 GiB within 105.28 GiB; native 262144, allocated prompt/reply window 262144, cache slots 262151
    vision: image and video input, a 0.84 GiB tower with 1.25 GiB of workspace reserved
    3 streams of 262144 prompt/reply tokens (4799 MiB a stream), eager; int8 KV cache (fp16 scale per 32 values)

The flags this rig runs, `EXTRA_ARGS` in `.env` plus one environment variable (load 383.7 s):

    EXTRA_ARGS="--parallel 3 --context 262144 --thinking --reasoning-effort xhigh --vision --max-tokens 32768 --kv-dtype int8 --mtp-confidence 0.60"
    TENSORFOLD_FACES_FP8=all

- `--parallel 3 --context 262144` — three lanes of the full native 262144 window on one rank (Flash Next, one
  rank). 262144 is the checkpoint's native window and it is granted at three lanes; asking for it at four is
  refused with `estimated largest fitting prompt-plus-reply window: 246909 tokens`. A round is weight-bound, so the
  lanes share its reads: three concurrent streams sustain **111.6 tok/s** aggregate against **42.1** for one —
  **2.65x**. Read the startup line's *allocated* window, not the flag you passed; with more than one lane a window
  can be silently clamped.
- `--kv-dtype int8` — halves the cache against BF16 and is what makes 262144 fit at three lanes. `bf16` at this
  window and lane count is refused. `int4` was measured and rejected: it needs a dequantisation per read, costing
  **−17%** on a 20k-context decode (41.0 vs 49.4 tok/s) and −1.6% aggregate, to buy capacity the traffic does not
  need.
- `--mtp-confidence 0.60` — the draft chain stops where the head is unsure. Swept 0.30 → 0.45 → 0.60 → 0.75 on the
  same bench: **80.6 → 84.9 → 88.9 → 88.9** tok/s aggregate at three lanes, while a long-reply decode *broke* at
  the top step (**34.4 → 32.9** on a 1024-token reply, both passes agreeing). 0.60 is the last value that still
  gains.
- `--thinking --reasoning-effort xhigh` — the replies run long on reasoning and the text streams as
  `reasoning_content` while `content` stays null until the budget is spent. The single-stream measurement below
  is a 772-token reply of which 748 tokens were reasoning.
- `--max-tokens 32768` — the server-side default, and the flag that matters once `--thinking` is on: a small
  client budget is spent entirely on `reasoning_tokens` and yields no `content` at all (see Measuring speed). A
  client that sends no `max_tokens` gets this default, and content returns normally.
- `--vision` — image input through the tower, and the reason the lane count is not lower: the engine refuses
  image input below two lanes (it shares a round's reads, see above). It takes inline base64 images; public
  HTTP(S) URLs need `--vision-urls` on top, which is deliberately not set. The tower costs 0.84 GiB and
  reserves 1.25 GiB of workspace.
- `TENSORFOLD_FACES_FP8=all` — loads BF16 faces as an e4m3 copy a round reads instead of the stored rows; a prompt
  keeps the rows, so the copy is a decode lane, never a prefill one. `1` covers the DeltaNet and attention linears,
  the ones a round re-reads most; `all` covers every BF16 face. Going from `1` to `all` took the three-lane
  aggregate **88.9 → 111.6 tok/s (+25.5%)** and a short-reply decode **34.4 → 43.9**, prefill and TTFT unmoved, and
  loosened the four-client queue: a request that waited 19 s behind three others now answers in 8.1 s.

  **Read the trade before keeping it.** Upstream's hard rule for speed work is that no precision is traded for
  speed — every token from the stored weights' own values — and this copy answers the stored rows only within 15%.
  The switch's own author measured the drafts' head accepting 48% where it accepted 64%, with the end-to-end number
  "not moving" on his branch, which is why `all` is not the default upstream. This rig measured **+25.5% end to
  end** instead, on 0.5.0 with the three rebased PRs and `--mtp-confidence 0.60`. The two measurements disagree and
  this file does not resolve the disagreement: the draft acceptance rate is the number that decides it and it was
  not read here. Sweep in [docs/engine-status.md](docs/engine-status.md).
- **The conversation cache** — not a flag, already on. A prompt that *strictly extends* a cached prefix returns
  `cached_tokens=19,862` and cuts the wall from **12.2 s to 0.2 s**; an *identical* re-ask and a shared system
  block both return `cached_tokens=0`. So the win lands on real chat traffic — every turn after the first. It
  serves **one conversation at a time**: four interleaved conversations never hit, whatever `--checkpoint-slots`
  says (tested 32 and 200), because the GDN state cannot truncate. See "The cache" below.

Port **8083** here (matches the hermes-agent / historical Spark OpenAI endpoint). The vLLM sibling also used
8083 — only one of the two fits in memory at a time.

The preflight figures, the memory budget the load fits in, and the earlier serving levers (`--ssd-experts`,
`--ple-on-ssd`, the MTP draft window) are in [docs/engine-status.md](docs/engine-status.md).

## Endpoint

```bash
curl -fsS http://127.0.0.1:8083/health
curl -fsS http://127.0.0.1:8083/v1/models
curl -fsS http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

## Status of the engine under test

What the rig measures today — TensorFold 0.5.0 with the three local changes, `--thinking`, three lanes at the
full 262144 window, int8 KV, `--mtp-confidence 0.60`, 8-bit faces on every layer:

- **Long-prompt context (15,460 prompt tokens):** TTFT **9.28 s**, prefill **1666 tok/s**, decode **51.5 tok/s**
on a short reply.
- **Aggregate decode, three lanes:** 1 stream **42.1** / 2 streams **82.0** / 3 streams **111.6 tok/s** —
  **2.65x** one stream. Against the untouched engine's **80.8** at the same three-client bench, the two accepted
  levers are worth **+38%**.
- **The conversation cache:** one conversation, first turn 12.17 s, every later turn **0.20-0.25 s**.

A measurement rule this rig learned the hard way: the **first pass after a load is cold** and must be discarded.
Every configuration measured here — including the untouched reference — showed a depressed prefill on the first
pass (1172-1503 tok/s) against a stable second pass (1644-1674). Undeclared, that transient masquerades as the
effect of whatever flag was just added.

### The cache

The engine caches conversation prefixes and the win lands on real chat traffic, but only under one pattern. A
request that *strictly extends* a cached prefix returns `cached_tokens=19,862` and answers in **0.2 s** where the
cold call took **12.2 s**. Two patterns that look equivalent return `cached_tokens=0`: re-asking the *identical*
prompt (12.3 s) and two requests sharing a long *system* block (17.8 s). A cache probe built on repeated identical
prompts will therefore conclude the cache is unimplemented — the wrong verdict, and one this rig published before
correcting it.

It is **per lane, and the lane count is the limit**. Measured with `--parallel 3` and a barrier synchronising the
sends: three conversations running *simultaneously*, one request in flight each, all hit on turns 2 and 3 —
`cached_tokens=3476`, **0.42 s** wall each — while their three cold first turns took **6.68 s** apiece, three times a
single cold prefill and therefore proof that the three requests really did occupy three lanes. Four or six
conversations *rotating* over those same three lanes hit **never**, at 32 checkpoint slots and again at 200.

So the working rule is **up to `--parallel` concurrent conversations each keep their own prefix**, and beyond that
they evict one another: at three lanes, a fourth conversation degrades all of them. Read the earlier claim in this
section — "one conversation at a time" — as the artefact of probing with more conversations than there were lanes.
The code's note still explains the shape of it: "LRU caches require strict-prefix hits because GDN state cannot
truncate". `--spill-gib` and `--snapshot-dir` are plumbed on CUDA (`checkpoint_slots`
reaches `CheckpointStore` in `server/app.py`) and the recipe already points snapshots at the persistent `/state`
bind; neither moved a number in this sweep.

Caveats:

- With thinking on, the decode numbers above are mostly reasoning tokens; the reasoning figure (748 of 772) is
  stated wherever a single stream is quoted.
- `scripts/bench.sh` reports `None` for TTFT and decode against this server, because its 1024-token budget is
  consumed entirely by `reasoning_tokens` and no `content` delta is emitted, so the client's clock never sees a
  first token. That is the bench against a thinking server, not a server defect. With no `max_tokens` at all the
  server returns content normally — **114 tokens, 78 of them reasoning** — because its `--max-tokens 32768`
  default is in force, and `max_tokens` 1024 or 4096 stop on their own with identical output.

The full history — the 0.3.6.3 baseline tables, the draft-window and 8-bit-faces sweeps, the load and preflight
figures — is in [docs/engine-status.md](docs/engine-status.md).

## Measuring speed

`scripts/bench.sh` (options: `TOKENS=256 scripts/bench.sh`, `EXTRA_BENCH_ARGS="--rounds 3 --json bench/last.json"`)
drives three workloads through the client's own clock:

- **prose** — a short French prompt asking for a 400-word essay: decode-bound.
- **code** — a short prompt asking for a Python function with docstring and pytest cases: decode-bound, code distribution.
- **prefill** — ~2,400 tokens of pasted context plus a one-line question: TTFT at scale.

Each workload runs twice: the first round pays for whatever the prefix cache does not hold, the second shows
the warm number. Every round makes one streaming request (TTFT and the inter-token rhythm) and one
non-streaming request (the server's own `usage`, which is where the token counts come from). Reported:
TTFT, `(completion_tokens - 1) / (stream total - TTFT)` for decode, `prompt_tokens / TTFT` for prefill (the
first token's decode sits inside TTFT, so it is a slight underestimate), and the delta count against
`completion_tokens` so the streaming rhythm can be sanity-checked.

**Against a thinking server the bench reports `None` for TTFT and decode:** a 1024-token budget is spent
entirely on `reasoning_tokens`, so no `content` delta is emitted and the client sees no first token. Read
`reasoning_content` as well, raise `max_tokens`, or point the bench at a `--no-thinking` server.

## Troubleshooting

- **`serve` exits 1 with `--ple-on-ssd reads the MLX checkpoint's n-gram shards from disk; an NVFP4 checkpoint's
  tables stay memory-mapped, so drop --ple-on-ssd`** — 0.3.6.3 refuses the flag for this checkpoint and exits
  before the weights load, so a profile tuned before that (this rig's `EXTRA_ARGS` carried it) will not start
  at all. Drop it: the tables are memory-mapped either way here.
- **`KeyError: ...ngram_embedding.shard_0.weight`** — the checkpoint is incomplete. A repacked copy (48 layer
  files, no `embedding-*` files ~79 GB, as some local copies are) has no n-gram table: download the published
  revision with `scripts/pull.sh` and serve that.
- **`401`/`403` from Hugging Face** — set `HF_TOKEN` in `.env` (a read token); the scripts pass it through and
  never write it anywhere else.
- **`mlock` warnings on start** — the PLE tables are pinned when the memory-lock limit allows it; the run
  continues without the pin. `scripts/serve.sh` already raises the limit and adds `IPC_LOCK`.
- **Slow first token after a rebuild** — the kernels are JIT-compiled on first use; keep `STATE_DIR` across
  rebuilds.
- **Out of memory at load** — add `--ssd-experts`, then lower `--context` (`--ple-on-ssd` is refused for this
  checkpoint on 0.3.6.3; see above).
- **A request comes back with no `content`** — with `--thinking` on, a small `max_tokens` is spent on
  `reasoning_tokens`; raise it, or read `reasoning_content` (see Measuring speed).

## Layout

```
Dockerfile              image: NVIDIA PyTorch + TensorFold (pinned ref), weights never baked in
scripts/build.sh        docker build, then `tensorfold --version` and a branch-only import check
scripts/preflight.py    header-only startup estimate: sizes the checkpoint without loading it, prints the plan
scripts/pull.sh         resumable download into HF_DIR
scripts/serve.sh        foreground serve with the Spark's mounts, caps and flags
scripts/smoke.sh        health, model ids, one timed completion
scripts/bench.sh        prose, code and prefill speed against a running server
bench/speed.py          what bench.sh runs: streaming TTFT + the server's usage, two rounds a workload
scripts/stop.sh         stop this rig's containers
docs/engine-status.md   engine versions, the change history, and every benchmark table
.env.sample             copy to .env: paths, port, HF token, EXTRA_ARGS
```

Weights, caches and logs stay on the host; the image holds the engine only. `scripts/build.sh` prints the
installed `tensorfold --version` and imports `tensorfold.families.qwen4_exp.cuda.nvfp4`, a module that exists
only on this branch, so a silently wrong build fails at build time.

## License

MIT (this repository). TensorFold is MIT; the model weights keep their own license on Hugging Face.
