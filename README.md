# Swift 1.5 on TensorFold — one DGX Spark

A test rig that serves the Swift 1.5 NVFP4 checkpoint on a single DGX Spark (GB10, 128 GB unified memory)
with [TensorFold](https://github.com/ashhart/TensorFold) 0.6.1: FP4 routed experts, BF16 everywhere else, and
the PLE layer's n-gram table in the BF16 layout this revision publishes.

The sibling rig [`Qwen3.8-Flash-Next-Single-DGX-Spark`](https://github.com/tournierjc/Qwen3.8-Flash-Next-Single-DGX-Spark)
serves the same model family with a patched vLLM; this one runs the TensorFold CUDA route end to end on the
Spark. The image carries the vision path: `transformers==5.17.0`, `av`, Pillow and the `qwen4_exp` frontend.
Upstream 0.6.1 serves **images** on this family now — the same MiaAI-Lab port this rig had carried since 0.6.0 —
so what the branch adds there is **video**: a clip is sampled at 2 fps, bounded to 256 frames, and its frame
groups ride the image path as extra placeholder blocks. Vision is dormant unless `--vision` is passed, and it
needs at least two lanes. Verified end to end: a solid red square and a solid blue square answered "Rouge"
(2.7 s) and "Bleu" (2.9 s).

## What it runs

| | |
| --- | --- |
| Model | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`, revision `3ff05202` — 186.4 GB over 296,474 tensors |
| Engine | TensorFold **0.6.1** (`ashhart/TensorFold@17c73e1`), served from the fork branch `integration/0.6.1` — the `Dockerfile` pins that branch's commit, override with `TF_REF`. On top of 0.6.1 it carries five changes, one commit each: the 8-bit projection copies gated by `TENSORFOLD_FACES_FP8`, **video** input on the vision frontend (upstream 0.6.1 carries the image half of that port), the 80,014-id MTP draft vocabulary, the PLE row prefetch, and a dormant 12-bit decode-face prototype (`TENSORFOLD_FACES_12BIT`). Everything the rig used to carry and upstream has since merged is gone from the branch: the `qwen4_exp` image port, the multi-row item-16 pair path, the fp32 reduce / `_fp4mm` block |
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

The engine the image serves is the fork's `integration/0.6.1` branch, and that is now the `Dockerfile`'s own
default (`TF_REPO`) — upstream 0.6.1 plus the rig's five changes, none of which upstream carries. Build it with
no arguments. To test another revision, pass both: `TF_REPO` alone is not enough, since without `TF_REF` the build
resolves the fork's `main` (upstream 0.6.1: no video, no 8-bit faces, and the unreduced 79,591-id draft
vocabulary) and fails its own build assertions.

```bash
scripts/build.sh                                                        # the pinned integration branch
TF_REF=<sha|branch> scripts/build.sh                                    # another commit of the same fork
TF_REPO=https://github.com/ashhart/TensorFold.git TF_REF=v0.6.1 scripts/build.sh   # upstream, for a baseline
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

What the rig last measured — TensorFold **0.5.0** with the three changes that are now the pinned branch's own
first commits, `--thinking`, three lanes at the full 262144 window, int8 KV, `--mtp-confidence 0.60`, 8-bit
faces on every layer. These figures have **not** been re-taken on 0.6.x: the rebase replays these paths
unchanged, but 0.6.1 brings 54 upstream commits over 0.6.0's tip, so treat the numbers below as the last
reading rather than as the current revision's.

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

## MTP draft vocabulary

The draft head does not score the checkpoint's whole vocabulary. It scores a list of token ids the engine
reads from `families/qwen4_exp/cuda/draft_vocab.txt` (`draft_token_ids("default")`; the engine documents the
list in its `docs/recipes/qwen3.8-flash-next.md`). A token outside the list can never be drafted, and every
draft is verified against the model's own samples, so a missing token costs acceptance and never
correctness -- which makes the list a pure speed decision, and worth choosing deliberately.

The 80,014-id list this rig serves is **part of the pinned revision** — `integration/0.6.1` carries it as
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
installed `tensorfold --version` and imports the symbols that exist only on the pinned branch — video input
(`tensorfold.vision.videos`), the 8- and 12-bit face helpers (`bf16.py`) and the Flash Next CUDA vision frontend
— so a silently wrong build fails at build time. (`nvfp4`/`nvfp4_moe` were the old check and proved nothing:
upstream 0.6.1 ships both.)

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
`feat/vision-video` and its own commit in the pinned revision.

| Branch in `tournierjc/TensorFold` | Head | What it carries |
| --- | --- | --- |
| **`integration/0.6.1`** | `808767fd479c6bd8dbb2eb68f2a3537f75e6d520` | **the pinned revision**: 0.6.1 + the 8-bit copies + video input + the 80,014-id draft vocabulary + the PLE row prefetch + the 12-bit decode faces |
| `feat/vision-video` | `3056063e3b05899d2e8b84769bedac19832efed7` | video input alone, on 0.6.1 (`--vision`; the image half is upstream now) |
| `integration/0.6.0` | `c3fa14f4cdd2454d32326c6bc2845a71cb76e7b6` | the previously pinned revision, on 0.6.0's base: what the rig measured before this rebase |
| `feat/vision-qwen4-exp` | `286eaca99aecc79813b69182b29d40d060b3cb5c` | the whole vision port (images and video) on 0.6.0's base; its image half is upstream in 0.6.1, superseded by `feat/vision-video` |
| `feat/mtp-draft-vocab` | `153bf32ae56c8017acaf75f9fff3a801e8e6f002` | the 80,014-id MTP draft vocabulary (0.6.0 base) |
| `cursor/ple-row-prefetch-0ff8` | `c2c392ba467fa3d4a579fb58f5b806783d6a442d` | the PLE row prefetch: a round asks for its n-gram pages while the GPU drafts (0.6.0 base) |
| `cursor/lossless-12bit-faces-0ff8` | `3864881c3508c02a0093d8724d4256facd2692ac` | the 12-bit shared-exponent decode faces — lossless, `TENSORFOLD_FACES_12BIT`, dormant unless set (0.6.0 base) |
| `pr/host-table-rows-by-file` | `79aa2a06c7ca52ade88a479f33149fcda55906ae` | a round's few n-gram rows read by file over the pool, not one after another (#103, closed unmerged) |
| `main` | `17c73e189f5e6a5304cda7ea37f086f9c49b4788` | upstream 0.6.1, realigned |
| `cursor/lossless-12bit-faces-1ba6` | `410d2ffad389f5248cb5ac652714eb749823fc7a` | the same 12-bit prototype under an older spelling (`kind=` where `-0ff8` uses `face=`), on the pre-0.6.0 base: kept for the record, superseded |

`integration/0.6.1` is what the rig serves, one commit per change on top of 0.6.1; the branches above are the
same changes in reviewable units. The four per-unit branches still sit on 0.6.0's base — replaying them is what
the integration branch already did, and they are kept as the record of each unit rather than rebuilt for its own
sake. What belongs to this rig rather than to the engine stays here: the builder that produced the vocabulary
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
