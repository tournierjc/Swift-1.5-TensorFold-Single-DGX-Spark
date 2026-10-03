# Swift 1.5 on TensorFold — one DGX Spark

A test rig that serves the Swift 1.5 NVFP4 checkpoint on a single DGX Spark (GB10, 128 GB unified memory)
with [TensorFold](https://github.com/ashhart/TensorFold) 0.6.3: FP4 routed experts, BF16 everywhere else, and
the PLE layer's n-gram table in the BF16 layout this revision publishes.

The sibling rig [`Qwen3.8-Flash-Next-Single-DGX-Spark`](https://github.com/tournierjc/Qwen3.8-Flash-Next-Single-DGX-Spark)
serves the same model family with a patched vLLM; this one runs the TensorFold CUDA route end to end on the
Spark. The image carries the vision path: `transformers==5.17.0`, `av`, Pillow and the `qwen4_exp` frontend.
Upstream serves **images** on this family since 0.6.1 and **video** since 0.6.3, so the whole vision port this
rig had carried since 0.6.0 is upstream and the rig's `feat/vision-video` branch is superseded: a clip is sampled
at 2 fps, bounded to 256 frames, and its frame groups ride the image path as extra placeholder blocks. Vision is
dormant unless `--vision` is passed, and it needs at least two lanes. Verified end to end on the pinned 0.6.3
revision: a solid red square and a solid blue square answered "Rouge" (2.6 s) and "Bleu" (1.3 s), and a solid red
clip answered "Rouge" (2.4 s) — video rides upstream's own path now.

## What it runs

| | |
| --- | --- |
| Model | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`, revision `3ff05202` — 186.4 GB over 296,474 tensors |
| Engine | TensorFold **0.6.2** (`ashhart/TensorFold@56e2e3e`), served from the fork branch `integration/0.6.2` — the `Dockerfile` pins that branch's commit, override with `TF_REF`. On top of 0.6.2 it carries the same five changes it carried on 0.6.1, one commit each: the 8-bit projection copies gated by `TENSORFOLD_FACES_FP8`, **video** input on the vision frontend (upstream carries the image half of that port), the 80,014-id MTP draft vocabulary, the PLE row prefetch, and a dormant 12-bit decode-face prototype (`TENSORFOLD_FACES_12BIT`). Everything the rig used to carry and upstream has since merged is gone from the branch: the `qwen4_exp` image port, the multi-row item-16 pair path, the fp32 reduce / `_fp4mm` block |
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

The engine the image serves is the fork's `integration/0.6.3` branch, and that is the `Dockerfile`'s own
default (`TF_REPO`/`TF_REF`) — upstream 0.6.3 plus the rig's two changes, neither of which upstream carries.
Build it with no arguments. To test another revision, pass both: `TF_REPO` alone is not enough, since without
`TF_REF` the build resolves the fork's `main` (upstream 0.6.3: no 8-bit faces, and the unreduced 79,591-id draft
vocabulary) and fails its own build assertions.

```bash
scripts/build.sh                                                        # the pinned integration branch
TF_REF=<sha|branch> scripts/build.sh                                    # another commit of the same fork
TF_REPO=https://github.com/ashhart/TensorFold.git TF_REF=v0.6.2 scripts/build.sh   # upstream, for a baseline
```

Logs go to the terminal that runs `scripts/serve.sh`; `scripts/stop.sh` stops a detached run and an
interrupted download.

## Serving on 128 GB

`scripts/preflight.py` prints the whole plan from the checkpoint's headers alone, in seconds, before any load.
It sizes the dense geometry and the window, and it takes no `--vision` flag: it does not add the tower's weights
or the `TENSORFOLD_VISION_WORKSPACE_MIB` reserve that a `--vision` serve counts in the line below, so read its
receipt as the non-vision plan. It also cannot run while a serve holds the memory (the second load is refused
outright: `estimated largest fitting prompt-plus-reply window: 0 tokens`).

The 0.6.3 serve reports these lines, and the whole load took **208.5 s** on its first start -- the revision's
kernel extensions were being compiled (`prompt kernels warmed in 99.8s`) -- with warm starts reusing those caches
(**115.3 s** on 0.6.2, the arm that was restarted warm; 0.6.1 measured the same in its own session):

    startup estimate 98.31 GiB within 104.95 GiB; native 262144, allocated prompt/reply window 262144, cache slots 262151
    the 95.4 GiB of mapped tables do not fit beside the weights and caches: lookups will page them from disk, which slows prompts (free memory to keep them resident)
    vision: image and video input, a 0.84 GiB tower with 1.25 GiB of workspace reserved
    Flash Next on CUDA: 1 to 6 MTP drafts a round, a chain stops before a later draft under 60%; up to 3 streams, each growing to 262144 prompt/reply tokens while memory lasts (22.6 GiB free for their caches, 4.47 GiB for one at the full window), eager; int8 KV cache (fp16 scale per 32 values); n-gram tables read in 25.3s; 0 decode graphs captured; idle prompt pieces 2048 rows; prompt kernels warmed in 3.7s

The n-gram tables cost 25-26 s on every start (25.1 s on the 0.6.1 arm, 25.3 s here) and the 186 GB of weights
are read once, so a restart on a warm cache is under two minutes.

The second line is not about this revision: the engine has printed it whenever the PLE tables are larger than
what is left beside the weights and the lanes' caches (`docs/engine-status.md` records it for 0.3.6.3 too), and
it means their lookups walk the page cache instead of staying resident on their own. **Read that first number as
the moment's, not the revision's**: 105.28 GiB on 0.6.0, 104.44 GiB at the 0.6.1 deploy, 102.95 GiB on the 0.6.1
arm served minutes before this one and 104.95 GiB on this arm — three sibling containers (`hermes-agent`,
`scalesync`, `captcha-solver`) hold memory on this host and their footprint moves between serves.
`--ple-on-ssd` is the lever if that ever costs prompt speed; the numbers below are what it costs today.

The flags this rig runs, `PARALLEL`, `CONTEXT` and `EXTRA_ARGS` in `.env` plus two environment variables:

    PARALLEL=3
    CONTEXT=262144
    EXTRA_ARGS="--thinking --reasoning-effort xhigh --vision --max-tokens 32768 --kv-dtype int8 --mtp-confidence 0.60"
    TENSORFOLD_FACES_FP8=all
    TENSORFOLD_VISION_WORKSPACE_MIB=1280

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
- `--vision` — image and **video** input through the tower, and the reason the lane count is not lower: the
  engine refuses image input below two lanes (it shares a round's reads, see above). It takes inline base64
  media; public HTTP(S) URLs need `--vision-urls` on top, which is deliberately not set. The tower costs
  0.84 GiB and reserves 1.25 GiB of workspace (`TENSORFOLD_VISION_WORKSPACE_MIB`; the engine's own default is
  4096 through 0.6.3, three times this rig's measured peak). A clip is decoded with PyAV, sampled at 2 fps up to
  256 frames, resized into the tower's grid, and sent as one timestamped placeholder block per frame group — the
  model reads the timestamps (`verified: a 2 s all-red clip answered "the frames show a solid red color across
  all frames (0.0s, 1.0s, 2.0s)"` in 1.65 s). Upstream has no video (0.6.1 and 0.6.2 alike): this half of the
  vision work is the branch's own.
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

### Measured on the pinned 0.6.3 revision

The rig was rebuilt on `integration/0.6.3` (`f5de97f`, image `43be03b19d69`) and redeployed with the same
`.env`, then run through `scripts/bench-suite.sh` twice, because a single pass is not a measurement. The
baseline is the 0.6.2 pass recorded below (35.0 / 57.2 / 73.2 aggregate), and the two 0.6.3 passes bracket it:

| Arm | Image | Load | 1 client | 2 clients | 3 clients |
| --- | --- | --- | --- | --- | --- |
| `integration/0.6.2` (`d26e09f`) | `b3677c5c500b` | 210.6 s | 35.0 tok/s | 57.2 tok/s | 73.2 tok/s |
| `integration/0.6.3` (`f5de97f`) pass A | `43be03b19d69` | 208.5 s | 32.9 tok/s | 55.4 tok/s | 70.6 tok/s |
| `integration/0.6.3` (`f5de97f`) pass B | `43be03b19d69` | 208.5 s | 34.9 tok/s | 57.3 tok/s | 73.2 tok/s |

Pass B agrees with the 0.6.2 baseline to 0.3%, and pass A is 3-6% under it: that spread is the run-to-run
variation this rig has always shown on the aggregate (the 0.6.2 pass recorded below had a polluted 2-client
reading of 19.5 tok/s against its own 57.2). **0.6.3 costs nothing measurable here** — the client-side speed is
unchanged too (prose 13.58 s against 13.55, code 10.40 against 10.48, prefill 1.53 against 1.54), and vision
passed on both: red image "Rouge", red clip "Rouge", blue image "Bleu".

### Measured on the pinned 0.6.2 revision

The rig was rebuilt and redeployed on `integration/0.6.2` and measured against the pin it replaces, both arms
served fresh on the same box from the same `.env` in one session, with one bench for both:
`scripts/bench-suite.sh <arm>` — a discard pass (the rig's rule: the first pass after a load is cold), then
`bench/aggregate.py` for N clients (**distinct** prompts — three clients sending the same prefix have the lanes
fight over one cached prefix, which alone cost 5 tok/s at three lanes — 1024-token replies), `bench/speed.py`
for one client, and `bench/vision_probe.py`, all run from inside the image, with `--thinking` on.

| arm | image | load (warm) | 1 client | 2 clients | 3 clients |
| --- | --- | ---: | ---: | ---: | ---: |
| `integration/0.6.1` (`808767f`) | `swift-tensorfold:061` | 115.3 s | 34.9 | 57.2 | 72.9 tok/s |
| `integration/0.6.2` (`d26e09f`) | `swift-tensorfold:local` | 115.3 s | 35.0 | 57.2 | 73.2 tok/s |

The 0.6.2 row is the second of three passes, all agreeing: 35.0 / 57.2 / 73.2, then 35.2 / 57.5 / 73.6, and a
first pass whose **two-client point read 19.5 tok/s** and is published here rather than dropped. That pass was
polluted: the serve log (0.6.2 prints a line a request) shows a **71,004-token, `finish=tool_calls` request
whose prefill took 55.09 s** interleaved between the one- and two-client runs, followed by three more turns on a
~72k context — the host's own `hermes-agent` container, whose local model provider is this same `:8083`
endpoint, holding a conversation while the bench ran. The 2-client point sits between two of those, which is
what the 9.8 / 10.4 tok/s per client is. **Nothing else on the box talks to `:8083`, and a bench here has to
check that it does not**: the per-request lines name the prompt size, the finish and the rate, and are how this
was found.

- **Load:** 210.6 s on the revision's first start (kernel extensions compiled for it: `warmed in 98.2s`),
  **115.3 s** warm — the same 115.3 s the 0.6.1 arm measured, so the rebase costs nothing to start.
- **Prefill:** a 2315-token prompt in 1.54 s, with `cached_tokens=2314` — this rig's prompts repeat, so that is
  a cached prefill, not a cold one. A cold 15,460-token prompt measured 1666 tok/s on 0.5.0.
- **Decode, one client:** **35.0 tok/s** on a 1024-token reply. `speed.py`'s 512-token replies: prose **37.8**,
  code **49.0** tok/s (`speed.py` reports `TTFT None` on this profile: with `--thinking` the stream's deltas are
  `reasoning_content`, and the script times `delta.content`).
- **Aggregate, N clients:** 1 → **35.0**, 2 → **57.2**, 3 → **73.2 tok/s** (2.09x one client), lanes within
  ~40% of each other (24.4 / 28.5 / 33.7 at three).
- **Vision, end to end** (`bench/vision_probe.py`, which generates what it asks about and takes the colour as an
  argument, so the answer has to come from the pixels): a 512x512 solid image answered `Rouge` in **2.50 s**
  (red) and `Bleu` in **1.28 s** (blue); a 2 s clip answered `Rouge` in **2.26 s**; a 2048x2048 image (4M
  pixels, 4172 prompt tokens) answered `Rouge` in **7.32 s** on 0.6.1. Draft acceptance over the whole suite:
  **7,074 of 10,745 drafts (65.8%)**, against 5,973 of 9,251 (64.6%) on the 0.6.1 arm.

**Against the previous pin, same box, same bench, same session** — `swift-tensorfold:061`
(`integration/0.6.1`, `808767f`, image `fa5ff1616f41`) against `swift-tensorfold:local` (`integration/0.6.2`,
`d26e09f`, image `b3677c5c500b`), both loaded from the same `.env`: **34.9 / 57.2 / 72.9** against
**35.0 / 57.2 / 73.2 tok/s** at one, two and three clients, `speed.py` within 0.04 s a workload and the vision
probes within 0.04 s a case. The rebase is free at this bench's resolution. The 0.5.0-era figure of **111.6
tok/s** at three lanes quoted below was taken with a different bench (longer replies, a different prompt set):
it is not comparable to the numbers above, and these two arms are what says the rebase did not cost speed.

### The 0.5.0 readings this rig still quotes

TensorFold **0.5.0** with the three changes that are now the pinned branch's first commits, `--thinking`,
three lanes at the full 262144 window, int8 KV, `--mtp-confidence 0.60`, 8-bit faces on every layer. These
figures predate 0.6.0 and were not re-taken on it: the rebase replays these paths unchanged, but 0.6.1 brings
54 upstream commits over 0.6.0's tip and 0.6.2 brings 12 more over that, so read them as the earlier revision's.

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

`scripts/bench-suite.sh <arm>` is the whole instrument, and the way a pin is measured here: it serves nothing
(point it at a running server), runs a **discard pass** first — the first pass after a load is cold, and this
rig's figures were learned by throwing one away — then `bench/aggregate.py` at 1, 2 and 3 clients,
`bench/speed.py` and `bench/vision_probe.py`, and writes every line to `bench/<arm>.log` along with the id of
the image **actually serving** (`docker inspect`: two revisions of one engine print the same startup lines, so
the tag is not evidence) and the server's own counters. Both arms of a comparison get the same suite, from the
same `.env`, served fresh back to back:

```bash
IMAGE=swift-tensorfold:061 scripts/serve.sh && scripts/bench-suite.sh before-061
docker rm -f swift-tensorfold
IMAGE=swift-tensorfold:local scripts/serve.sh && scripts/bench-suite.sh after-062
```

**Check who else is on the endpoint before believing a number.** The host's own `hermes-agent` container points
its local model provider at this same `:8083`, and a 71k-token agent turn landing inside a two-client run took
that point from 57 to 19.5 tok/s. The serve log is what shows it — 0.6.2 prints a line a request, with the
prompt size, the finish and the rate, where 0.6.1 leaves the `/health` counters. Repeat a point that disagrees
instead of publishing it or dropping it.

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

## MTP draft vocabulary

The draft head does not score the checkpoint's whole vocabulary. It scores a list of token ids the engine
reads from `families/qwen4_exp/cuda/draft_vocab.txt` (`draft_token_ids("default")`; the engine documents the
list in its `docs/recipes/qwen3.8-flash-next.md`). A token outside the list can never be drafted, and every
draft is verified against the model's own samples, so a missing token costs acceptance and never
correctness -- which makes the list a pure speed decision, and worth choosing deliberately.

The 80,014-id list this rig serves is **part of the pinned revision** — `integration/0.6.2` carries it as
`families/qwen4_exp/cuda/draft_vocab.txt`, and the Dockerfile fails the build unless the installed package has
it with that count (a build against upstream or an older tag stops there in seconds instead of scoring 79,591
ids at run time, the failure mode that cost the sibling recipe ~17%). **80,014 ids** — the engine's shipped
79,591-id list whole, plus **423** ids ranked by frequency over the engine's source, its tests and the CPython
stdlib beside it, with the byte-fallback range pinned (ids 0-255, what BPE falls back to for accents, CJK and
emoji). Being a strict superset of the engine's list, it cannot lower coverage on any text: 0 of the 79,591 ids
are missing, checked against the list the built image installs
(`88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e`, the digest the engine's own
`docs/recipes/qwen3.8-flash-next.md` publishes). This file's own digest is
`8facf56e11ad522ca8ba1d396755b6ce7cc98f2bf226498780fcc7806231c192`, asserted by `tests/test_draft_vocab.py`.

**Credit.** The construction is ported from MIA AI Lab's reduced-vocabulary MTP drafting,
["mia's recipe"](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) --
`files/build_draft_vocab.py`, `files/build_draft_vocab_extend.py` and its vLLM wiring
`files/patch_mtp_draft_vocab.py` (AGPL-3.0-or-later, Copyright (C) 2026
[MiaAI Lab](https://x.com/MiaAI_lab)). `scripts/build_draft_vocab.py` here is this rig's adaptation for the
TensorFold `qwen4_exp` CUDA engine, and keeps the three rules that script learned: a `--base` list is a floor
and is never lost to a ranking; the byte-fallback range is pinned whatever its frequency; and only real text
adds ids, by frequency.

Coverage on held-out slices, measured with `scripts/build_draft_vocab.py` (nothing in these slices was
ranked on; the last row is the replies the rig's own bench logs quote):

| Held-out slice | Tokens | Engine's list (79,591) | This rig (80,014) | Ranked only, no floor (32,905) |
| --- | ---: | ---: | ---: | ---: |
| engine `docs/**/*.md` | 32,752 | 99.4046% | **99.9084%** | 99.1268% |
| engine `README.md` | 5,294 | 99.4333% | **99.9244%** | 99.2255% |
| engine `CHANGELOG.md` | 2,823 | 99.3270% | **99.8229%** | 98.8665% |
| this rig's `README.md` | 3,280 | 99.6037% | **99.8780%** | 97.7134% |
| this rig's `scripts/*.sh` | 2,940 | 99.4898% | **99.8639%** | 93.5374% |
| mia's recipe `README.md` | 6,874 | 99.4908% | **99.8109%** | 97.4833% |
| the model's own output | 575 | 99.3043% | 99.3043% | 94.9565% |

The last column is why the engine's list is kept as the floor: dropping it to shrink the draft head costs 0.5
to 6 points of coverage on every slice.

Decode rate was measured too, with the full 186 GB checkpoint loaded, one list against the other: it came out
**neutral** -- prose 35.4 tok/s against the engine's list's 35.5, and code 49.5 against 49.6. The decoding win a
reduced list is worth was already banked in the engine's own 79,591 ids; the 423 this rig adds are kept for the
coverage that list leaves on the table, not for a rate, and they cost none.

To rebuild it, from the repository root (the image carries `tokenizers`; `HF_DIR` comes from `.env`). `--base`
is the list to extend: the engine's own shipped 79,591 ids, read from a checkout of the upstream revision under
`dev/` (the example spells it `dev/port`, beside the fork's `dev/repo`; this rig's own tests look for either
name, plus `dev/pinned`). The output goes to a scratch path, not into this repository — the list is a commit on
the engine branch now, so compare it (`sha256sum`; the rig's is `8facf56e…`) and commit it there:

```bash
TOK=$(echo /hf/hub/models--ukisai--Swift-1.5-Qwen3.8-Flash-Next-NVFP4/snapshots/*/tokenizer.json)
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python3 \
  -v "$PWD":/rig -v "$HF_DIR":/hf:ro -v /usr/lib/python3.12:/stdlib:ro swift-tensorfold:local -B \
  /rig/scripts/build_draft_vocab.py "$TOK" /tmp/draft_vocab.txt \
  --base /rig/dev/port/src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt --size 98304 \
  '/rig/dev/repo/src/**/*.py' '/rig/dev/repo/tests/**/*.py' '/stdlib/**/*.py'
```

Add the model's own output as `.jsonl` (a `text` field per line, `:N` to repeat it) to weight it against the
generic corpus; `--help` documents `--base`, `--byte-fallback-max`, `--keep-below`, `--min-count`,
`--holdout` and `--sweep`. `tests/test_draft_vocab.py` checks the builder's rules and the list the pinned
engine ships; `pytest.ini` keeps the engine checkouts under `dev/` out of that run.

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
Dockerfile              image: NVIDIA PyTorch + TensorFold (the fork's pinned integration commit)
scripts/build.sh        docker build, then `tensorfold --version` and a branch-only import check
scripts/preflight.py    header-only startup estimate: sizes the checkpoint without loading it, prints the plan
scripts/pull.sh         resumable download into HF_DIR
scripts/serve.sh        foreground serve with the Spark's mounts, caps and flags
scripts/smoke.sh        health, model ids, one timed completion
scripts/bench.sh        prose, code and prefill speed against a running server
bench/speed.py          what bench.sh runs: streaming TTFT + the server's usage, two rounds a workload
scripts/stop.sh         stop this rig's containers
docs/engine-status.md   engine versions, the change history, and every benchmark table
scripts/build_draft_vocab.py  builds the MTP draft head's reduced vocabulary (ported from mia's recipe)
tests/test_draft_vocab.py     the builder's rules and the vocabulary the pinned engine ships
pytest.ini              default test collection: this rig's tests/, not the engine checkouts under dev/
.env.sample             copy to .env: paths, port, HF token, EXTRA_ARGS
```

Weights, caches and logs stay on the host; the image holds the engine only. `scripts/build.sh` prints the
installed `tensorfold --version`, imports the one symbol that still exists only on the pinned branch — the 8-bit
face helper (`bf16.faces_8bit`) — and the `Dockerfile` asserts the pinned draft vocabulary by data, 80,014 ids.
Those two are the whole proof: video input (`tensorfold.vision.videos`) and the Flash Next CUDA vision frontend
are imported as smoke checks, but upstream 0.6.3 ships both, so they no longer say anything about the ref that
was built. The 12-bit faces are checked nowhere, because they are deliberately not in the pinned revision.
(`nvfp4`/`nvfp4_moe` were the original check and proved nothing: upstream ships both, in 0.6.1 and 0.6.2
alike.)

## Engine revision and the branches

Every engine change this rig serves lives on a branch of
[`tournierjc/TensorFold`](https://github.com/tournierjc/TensorFold) — a change lives on its branch there, never
as a file copied into this repository. The image installs the revision the Dockerfile pins
(`TF_REPO`/`TF_REF`) and overlays **nothing**: the revision carries the whole rig.

The rebase onto **upstream 0.6.0** (`c464617`) dropped what upstream had already merged: the multi-row item-16
pair path ([#102](https://github.com/ashhart/TensorFold/pull/102)) and the fp32 reduce / `_fp4mm` block change
([#105](https://github.com/ashhart/TensorFold/pull/105)), both merged in 0.6.0 with the same patch, so their
branches are gone. The 8-bit projection copies
([#104](https://github.com/ashhart/TensorFold/pull/104)) upstream declined (*no precision traded for speed*) and
this rig keeps: `TENSORFOLD_FACES_FP8=all` carries the rig's reference numbers.

The rebase onto **upstream 0.6.1** (`17c73e1`, 54 commits and 200 files over 0.6.0's tip) dropped one more of
them: **the `qwen4_exp` image port**. Upstream 0.6.1 serves images on this family with the same MiaAI-Lab
patch — `EncodedVision`, `vision_config` and `image_positions` are byte-identical, and upstream's own
`tests/cuda/test_flashnext_vision.py` asserts the names this port introduced (`pbuf.rope_rows`,
`st.rope_delta`). What stays on the branch is the video half of that work, now its own branch
`feat/vision-video` and its own commit in the pinned revision — both deleted on the 0.6.3 rebase, when
upstream merged video itself.

The rebase onto **upstream 0.6.2** (`56e2e3e`, 12 commits and 50 files) replayed the five changes with **no
conflict at all** — the diffstat of the five is identical to the one they had on 0.6.1 — because upstream's work
in this release lands elsewhere: Flash Next at 64k-128k *on Macs*, the 27B's GDN tree kernel and DFlash2 drafter
launch on CUDA, the GLM-5.3 checkpoint credit and its mixed-bit EXL3 refusal, and a set of server fixes (a line a
request and the done line on the CUDA server, the client-gone check past descriptor 1023, no `.partial.safetensors`
left by an interrupted snapshot write, one GPU-generation reading). None of the files the branch owns
(`families/qwen4_exp/cuda/*`, `vision/*`, `host_table.py`, `server/messages.py`, `server/prompts.py`) is among
them, and **none of the five is upstream in 0.6.2** — checked by symbol, not by subject:
`TENSORFOLD_FACES_FP8`, `TENSORFOLD_FACES_12BIT`, `TENSORFOLD_PLE_PREFETCH`, `vision/videos.py` and
`families/qwen4_exp/cuda/draft_vocab.txt` are still absent there. Two entries of the release matter to this rig
anyway. The config check now accepts an **FP8 n-gram table** in NVIDIA's MIXED_PRECISION Flash Next export
([#179](https://github.com/ashhart/TensorFold/pull/179)): this checkpoint's table is BF16 so nothing changes here,
but a MIXED_PRECISION sibling of it was refused outright on 0.6.1. And `--mtp-confidence` now defaults to
**0.70** upstream where this rig pins **0.60** explicitly (swept here: 0.60 is the last value that still gains).

| Branch in `tournierjc/TensorFold` | Head | What it carries |
| --- | --- | --- |
| **`integration/0.6.3`** | `f5de97f917b51735dec57459daad8bd736642cb9` | **the pinned revision**: 0.6.3 + the 8-bit projection copies + the 80,014-id draft vocabulary. Upstream 0.6.3 merged the vision port whole, so video is no longer a branch commit; the PLE row prefetch and the 12-bit faces are deliberately not carried here (the 12-bit branch is kept for future work, see below) |
| `integration/0.6.2` | `d26e09fbd723f73dba84b6c6c4c2b0816ce98c10` | the previously pinned revision: 0.6.2 + the 8-bit copies + video input + the 80,014-id draft vocabulary + the PLE row prefetch + the 12-bit decode faces |
| `integration/0.6.1` | `808767fd479c6bd8dbb2eb68f2a3537f75e6d520` | the previously pinned revision, on 0.6.1's base: what the rig measured before this rebase |
| `integration/0.6.0` | `c3fa14f4cdd2454d32326c6bc2845a71cb76e7b6` | the 0.6.0 rebase, two rebases back |
| `feat/vision-qwen4-exp` | `eda6477` | the 8-bit projection copies alone, rebased onto 0.6.3 (the branch used to carry the whole vision port; images went upstream in 0.6.1, video in 0.6.3) |
| `feat/mtp-draft-vocab` | `65aaf68` | the 80,014-id MTP draft vocabulary, rebased onto 0.6.3 (upstream still ships the unreduced 79,591-id list) |
| `cursor/ple-row-prefetch-0ff8` | `497ca8d` | the PLE row prefetch: a round asks for its n-gram pages while the GPU drafts — rebased onto 0.6.3, **not** integrated (measured no value) |
| `cursor/lossless-12bit-faces-0ff8` | `32ecbe5` | the 12-bit shared-exponent decode faces — lossless, bit-exact, 0.83x the stored bytes — rebased onto 0.6.3 and **kept for future improvements, not integrated**. A lossless *8-bit* face is impossible: sign+mantissa is already 8 bits, and measured over four real faces 0.00% of 32-deep K-groups share one exponent, so an 8-bit shared-exponent face escapes 100% of groups and costs more than the raw BF16 |
| `pr/host-table-rows-by-file` | `79aa2a06c7ca52ade88a479f33149fcda55906ae` | a round's few n-gram rows read by file over the pool, not one after another (#103, closed unmerged) |
| `main` | `9356df5c424b0c36b7737e37873a6f968b08de79` | upstream 0.6.3, realigned |
| `cursor/lossless-12bit-faces-1ba6` | `410d2ffad389f5248cb5ac652714eb749823fc7a` | the same 12-bit prototype under an older spelling (`kind=` where `-0ff8` uses `face=`), on the pre-0.6.0 base: kept for the record, superseded |

The rebase onto **upstream 0.6.3** (`9356df5`, 81 commits and 163 files over 0.6.2) changed what the branch
carries rather than how it carries it. Upstream merged the vision port whole — `588921b feat(vision cuda): Flash
Next takes video input`, plus the visual-token budget and the tower offload — so **video left the branch**: the
video commit was dropped in the replay, and `feat/vision-video` is superseded — its only remaining
delta was a stale 64 MiB clip cap against upstream's deliberate 16 MiB one — so the branch is deleted;
its head stays on the tag `backup/removed/feat_vision-video` (`3056063`). Two changes stay, both still
absent from 0.6.3 by symbol: the 8-bit faces (`TENSORFOLD_FACES_FP8` / `faces_8bit`) and the reduced
`draft_vocab.txt` (80,014 ids against upstream's 79,591). Two do not: the **PLE row prefetch** and the **12-bit
decode faces** are not integrated here. Both still rebase cleanly onto 0.6.3 and both branches are kept (the
12-bit one for future work), but neither earns its keep on this rig — the 12-bit face is lossless yet reads
0.83x the stored bytes against the e4m3 copy's 0.52x, and there is no lossless 8-bit to replace that copy with
(the measurement is in the branch table above).

`integration/0.6.3` is what the rig serves, one commit per change on top of 0.6.3; the branches above are the
same changes in reviewable units, all rebased onto the same release rather than left on 0.6.0's base. What belongs to this rig rather than to the engine stays here: the builder that produced the vocabulary
(`scripts/build_draft_vocab.py`), the checks that hold it to its invariants (`tests/test_draft_vocab.py`,
`pytest.ini`), and the sections above.

### Why nothing is overlaid any more

The Dockerfile used to copy `patch/draft_vocab.txt` over the installed package on every build, because the
vocabulary existed only as a file in this repository. It is a commit on the pinned revision now, byte for byte
the same file (`8facf56e…`), so the overlay carried no delta and is gone. What replaces it is an assertion in
the build: the installed package's own `families/qwen4_exp/cuda/draft_vocab.txt` must hold 80,014 sorted ids,
so pinning a revision without the port fails in seconds instead of scoring 79,591 ids at run time. The rule the
overlay taught still holds for any future delta: never overlay a whole working tree — that replaces every file
the tree lacks with the revision it happens to carry, which is how this image once shipped an older
`vision/config.py` and refused the checkpoint at launch.

The numbers each change is kept for are in the sections above; `docs/engine-status.md` holds the change
history.

## License

MIT (this repository). TensorFold is MIT; the model weights keep their own license on Hugging Face.

## Credits

- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and the TensorFold contributors (MIT) is
  the engine this rig serves, including the reduced-vocabulary MTP drafting path the list above feeds.
- The MTP draft vocabulary is built with a technique ported from **MIA AI Lab's** reduced-vocabulary MTP
  drafting, ["mia's recipe"](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  (AGPL-3.0-or-later, Copyright (C) 2026 [MiaAI Lab](https://x.com/MiaAI_lab)):
  `files/build_draft_vocab.py`, `files/build_draft_vocab_extend.py` and its vLLM wiring
  `files/patch_mtp_draft_vocab.py`. What is taken is the technique -- the corpus doctrine and its three rules
  -- re-expressed for this engine in `scripts/build_draft_vocab.py` and the pinned branch's
  `families/qwen4_exp/cuda/draft_vocab.txt`; no file from
  that repository is copied, and this repository stays MIT.
- The checkpoint is
  [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4);
  the weights keep their own license on Hugging Face.
