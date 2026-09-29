# Swift 1.5 on TensorFold — one DGX Spark

A test rig that serves the Swift 1.5 NVFP4 checkpoint on a single DGX Spark (GB10, 128 GB unified memory)
with [TensorFold](https://github.com/ashhart/TensorFold), the branch **`nvfp4-flash-next`**, which reads that
checkpoint as it ships: FP4 routed experts, BF16 everywhere else, and the PLE layer's n-gram table in the
BF16 layout this revision publishes.

The sibling rig [`Qwen3.8-Flash-Next-Single-DGX-Spark`](https://github.com/tournierjc/Qwen3.8-Flash-Next-Single-DGX-Spark)
serves the same model family with a patched vLLM; this one exists to run the TensorFold CUDA route end to end
on the Spark and to compare the two. Text only: the server refuses image, audio and video input, and the
checkpoint's vision tower is not read.

## What it runs

| | |
| --- | --- |
| Model | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`, revision `3ff05202` — 186.4 GB over 296,474 tensors |
| Engine | `ashhart/TensorFold@main`, pinned to `191188075bca56a7c71074a79375eb4c1cb22e1c` (0.3.6.3, override with `TF_REF`) |
| Upstream PR | [ashhart/TensorFold#67](https://github.com/ashhart/TensorFold/pull/67) (merged as 0.3.6.3) |
| Base image | `nvcr.io/nvidia/pytorch:26.07-py3` (36.5 GB as pulled here) — CUDA, torch 2.13, triton, the extension compiler |
| Endpoint | OpenAI-compatible on `:8083` (`/health`, `/v1/models`, `/v1/chat/completions`, streaming and tool calls) |
| Speed | `scripts/bench.sh` → `bench/speed.py`: TTFT, prefill rate and decode rate for prose, code and a long prefill |

Pinned commit (in the `Dockerfile` as `ARG TF_REF`): `191188075bca56a7c71074a79375eb4c1cb22e1c`.

**Quality:** ModelOpt NVFP4 dequant is `W = E2M1 * fp32(e4m3) * weight_scale_2` (no extra `2**-7`). This rig's own
pinned fix for that factor was `36a5bc4` on the fork; upstream fixed the same formula on top of the merge and
0.3.6.3 carries it, along with the grouped NVFP4 MoE that reads the routing plan on the GPU. Prior empty/`im_end`
loops came from the erroneous `2**-7` scale; with either fix the replies read as text.

On 0.3.6.3 the reply streams as `reasoning_content` and leaves `content` null until the thinking budget is spent,
so a client that reads only `delta.content` sees an empty stream, counts `deltas=0` and has no TTFT. `--no-thinking`
puts the text back in `content`; a benchmark that wants the first token's time should read either field.

The checkpoint's own `ple_embedding.ngram_embedding.shard_N.weight` tensors are BF16 `[2500012, 160]` rows with
no per-shard scales — 128 shards, 320,001,536 rows, 29.8 GiB, memory-mapped and gathered a lookup at a time.

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

Logs go to the terminal that runs `scripts/serve.sh`; `scripts/stop.sh` stops a detached run and an
interrupted download.

## Serving on 128 GB

The first `serve` compiles kernels (triton and torch extensions) into `STATE_DIR`; later starts reuse them.
`scripts/preflight.py` prints the whole plan from the checkpoint's headers alone, in seconds, before any load —
it runs the same estimate the engine runs first. Measured on this checkpoint (revision `3ff05202`): 97.39 GiB
within a 104.28 GiB budget, native window 262,144 tokens, weights 78.54 GiB, loading 18.85 GiB, cache
workspace 18.14 GiB, and the n-gram table at **95.37 GiB** — 102.4 GB over 128 shards in the BF16 layout this
revision ships, three times the 29.8 GiB the MLX 4-bit layout takes. It does not fit beside the weights and
caches, so its pages are read from disk during lookups; `--ple-on-ssd` makes that explicit instead of relying
on the page cache.

Then the levers, in `.env` as `EXTRA_ARGS`:

- `--ssd-experts 90` — stream routed experts into a 90 GiB GPU pool for models past the memory budget.
- `--mtp-drafts 5 --mtp-confidence 0.20 --no-thinking --context 8192` — what this rig serves: MTP for decode,
  answers in chat `content`, and an 8k window so unified memory has headroom for hermes-agent beside the
  weights; drafts `0` disables drafting. Five rather than ten because prose accepts about 2.5 drafts a round
  and code about 5.5 at the same round cost (table above); ten costs prose 3.5 tok/s and buys code nothing.
- `--ple-on-ssd` — refused for an NVFP4 checkpoint on 0.3.6.3 (`serve` exits 1 before the weights load; the
  tables stay memory-mapped here, see Troubleshooting). It applies to the MLX checkpoint's n-gram shards, where
  it is worth about 40 GiB at peak for a few percent of decode speed.
- `--parallel 2` — two requests decoded together, windows sharing each round's forward (Flash Next, one rank).
- `--context N` — prompt plus reply window; the CUDA default is the affordable native capacity.
- `--no-drafts` — the serial reference: same output, slower.

Port **8083** here (matches the hermes-agent / historical Spark OpenAI endpoint). The vLLM sibling also used 8083 — only one of the two fits in memory at a time.

## Endpoint

```bash
curl -fsS http://127.0.0.1:8083/health
curl -fsS http://127.0.0.1:8083/v1/models
curl -fsS http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"swift-1.5","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128}'
```

## Status of the engine under test

Built and checked on the DGX Spark this rig targets:

- The first end-to-end run found a blocker and it is fixed on the branch at `ba6cb76`: the startup memory
  estimate could not size the checkpoint's `F8_E4M3` block scales (73,728 tensors), so `serve` died in 15 s
  with `CUDA startup memory geometry could not be established on every rank: 'F8_E4M3'`. After the fix the
  estimate runs over the real headers and the load proceeds.
- `scripts/build.sh` → image `swift-tensorfold:local`, 36.5 GB; `tensorfold --version` = 0.3.6.2 (from the
  pinned commit), and the build-time import of `families.qwen4_exp.cuda.nvfp4` — a module that exists only on
  this branch — passed, so a wrong ref fails at build time.
- Dependencies inside the image: hf-hub 1.24.0, tokenizers 0.23.1, safetensors 0.8.0, jinja2 3.1.6, numpy 2.1.0,
  torch 2.13.0a0 (NVIDIA's 26.07 container).
- `tensorfold info ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` inside the image resolved the family from the
  published `config.json`: `qwen4_exp`, 48 layers, 512 experts, `quantization modelopt`, "runs on NVIDIA GPUs
  (CUDA)".
- Hugging Face answers the repository listing anonymously from the Spark (100 files, 31 `embedding-model-*`),
  so `scripts/pull.sh` needs no token today; keep `HF_TOKEN` in `.env` for the day that changes.

0.3.6.3's CUDA suite runs green in the same container this image is built from: **728 passed, 75 skipped,
0 failed** on a GB10 (the `nvfp4-flash-next` branch this rig pinned before it, 0.3.6.2: 460 passed, 70
skipped, 0 failed; with the reduce and `_fp4mm` changes below in, 728/75/0 again). Against the real published
weights, checked in-process:

- 16 routed experts bit-exact against the reference dequantization; FP4 matmul error 2.6e-3 / 2.3e-3; BF16
  faces 2.9e-3; row invariance from 1 to 64 rows; 512 MTP experts exact; the n-gram hashing constants equal
  value for value.
- The n-gram table on real bytes: shard 2 of `embedding-model-00006-of-00131.safetensors` gathers rows
  bit-identical to an independent read of the file, and 128 shards of that size make the 320,001,536-row
  table the digest derives.

**Measured on this rig on 0.3.6.3 (`1911880`), the complete 186 GB checkpoint, one DGX Spark:** the load runs
end to end — `97.39 GiB within 105.04 GiB`, a 262144-token window, 21 decode graphs captured, the 95.4 GiB of
mapped tables paged from disk, n-gram tables read in 291 s, loaded in 545 s. Greedy, TTFT from the first
streamed chunk (which on this checkpoint is `reasoning_content`, not `content`), decode as
`(completion - 1) / (total - TTFT)`, prefill as `prompt / TTFT`, two rounds each, identical across rounds:

| workload | prompt tok | decode | prefill | TTFT |
| --- | --- | --- | --- | --- |
| prose (400-word essay) | 37 | **27.2 tok/s** | — | 0.16 s |
| code (`merge_intervals` + pytest) | 59 | **54.6 tok/s** | — | 0.20 s |
| 2275 tokens of context, one-line question | 2275 | 32.9 tok/s | **1181 tok/s** | 1.93 s |

Against the `b4bf826` ref this rig had been serving, same checkpoint and profile: prose 17.9 → 27.2 tok/s,
code 27.6 → 54.6, prefill 798 → 1181 (the non-stream totals of `scripts/bench.sh` agree: 14.52 → 9.53 s,
9.62 → 4.86 s, 5.90 → 3.41 s). The targets this rig was pointed at are prose 30 / code 45 / prefill 1000:
**code and prefill clear them, prose reaches 91% of it.**

**The draft window this rig carries is the prose lever.** `--mtp-drafts 10` was tuned on the branch, where a
round cost less; on 0.3.6.3 the same window pays for a wider verification of drafts that prose does not
accept. Sweeping it on the warm server, same checkpoint and everything else fixed:

| `--mtp-drafts` | prose | code | prefill |
| --- | --- | --- | --- |
| 10 | 22.9 | 53.8 | 1175 |
| 5 | **26.4** | **54.5** | **1190** |
| 3 | 27.3 | 42.8 | 1117 |

Three drafts buy nothing over five on prose and cost code a fifth of its rate, so the rig now serves five.
Prose accepts roughly 2.5 drafts a round against code's 5.5, at the same round cost - which is why the
window that suits code over-pays on prose, and why the next gain here is the draft head's acceptance rather
than bandwidth.

`--mtp-confidence` is the other half of that trade-off, and it pays off without a cost: on five drafts, 0.30
against the 0.20 this rig carried gives prose 27.2 against 26.4 tok/s while code (54.6 against 54.5) and
prefill (1181 against 1190) do not move - a higher threshold stops a chain where the draft head is unsure,
which is what prose's drafts are, and leaves a confident chain alone. So the rig serves 5 drafts at 0.30.

Not verified yet, and worth reporting from a run here:

- The 95.4 GiB of n-gram tables do not fit beside the weights, so every lookup pages from disk. Measured on
  the served endpoint: decode reads **0.6-0.8 KiB per token from storage**, so the paging is not a throughput
  factor at this window. (`--ple-on-ssd` is refused for an NVFP4 checkpoint - `serve` exits 1 - and a smaller
  `--context` buys memory headroom without a measured effect on rate.)
- One of the 128 n-gram shards is proven against real bytes; the other 127 are read by the same code path and
  each shard's header is checked at load time, a mixed layout raising rather than loading wrongly.

**The dense BF16 faces are the round's other half, and an 8-bit copy of them pays.** This checkpoint quantizes
only the routed experts: a layer is 1200 MiB of NVFP4 experts and 150 MiB of e4m3 scales against **147.6 MiB
of BF16**, 110 MiB of it the DeltaNet and attention projections. A round verifies a handful of rows, so those
dense faces are re-read whole every round however few tokens come out of it - **6.9 GiB a round**, and 53% of
the round's device time in `_b16mm` (a full `block_n`/`bk`/`warps`/`stages` sweep at 7 rows buys 5%, so the
matmul is not mistuned: it is reading bytes). `TENSORFOLD_FACES_FP8=1` loads those projections with an 8-bit
copy a round reads instead of the rows (prompts keep the rows), and the same client then measures:

| `TENSORFOLD_FACES_FP8` | prose | code | prefill |
| --- | --- | --- | --- |
| off | 27.2 | 54.6 | 1181 |
| `1` | 32.2 | 63.6 | 1196 |
| `1`, with the 16-pair multi-row item below | **40.0** | **75.8** | **1486** |
| `4`, same | 38.5 | 73.5 | 1496 |

`4` copies the same projections affine group-of-32 (0.31x the stored bytes against 0.39x for e4m3) and is **not**
kept: fewer bytes, but the affine lane is slower a byte than the e4m3 one, and the round ends up 4% longer on
prose and 3% on code. The same reason `all` failed - on this chip the quantized lanes' efficiency does not scale
with the narrower format, so `1` is the end of that road and the bytes left are the ones that are already cheap.

The server's own round counters move with it: prose 76.4 -> 73.6 ms a round at 2.19 tokens, code 83.8 -> 76.7
ms at 4.57. `all` extends the copy to every BF16 face and was **not** kept - it cuts the round further, but
the router, hyper-connections and shared expert steer which experts run and how the streams mix, so the
drafts' head (calibrated to the BF16 body) accepts 48% where it accepted 64% and the end-to-end number does
not move. The lm_head's rows never get a copy for the same reason: the drafts' head is a quantized copy of
those very rows (0 of 63 drafts accepted once they were coarsened, replies garbled).

**The multi-row item holds 16 pairs, not 64, and that is worth 24%.** ``Plan`` gives each item 16 pairs when the
arithmetic is the decode form and 64 in the prefill form, and the kernel holds only those two. Measured at a
prompt's own row count (2275 x 10 pairs over 512 experts, `dev/moe_tile.py`): a 64-pair item reads 549 items of
the stack in **21.00 ms**, a 16-pair item reads 1788 items in **14.09 ms**. The fatter item buys 3x less traffic
that turns out to be L2 hits anyway (351 GB/s against the GB10's ~273 GB/s of DRAM), and pays for it with 3x
fewer items competing for the SMs. Since the verify window of a decoding round runs the same multi-row
arithmetic a prompt does, the change moves everything at once, served on the published checkpoint:

| item | prose | code | prefill |
| --- | --- | --- | --- |
| 64 pairs (upstream's default) | 32.2 | 63.6 | 1196 |
| **16 pairs** | **40.0** | **75.8** | **1486** |

`TF_REF=3ec1227 scripts/build.sh` puts the branch in the image and the image then serves the 16-pair row with
no mount at all: prose 40.1, code 75.9, prefill 1509 (the two rows above were taken through the mount and agree
within noise). **Verify the image by what it does, not by the build's exit code.** The first attempt here was a
silent no-op - `build.sh` sourced `.env` *after* the caller's variables, so `.env`'s pinned ref won and the
image kept serving 27.2/54.5/1180, byte-identical to the released package. The script now lets the caller win,
and `direct_url.json` in the image names the commit it was built from.

**The 4-bit checkpoint is the remaining reference, measured here.** Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
(4-bit throughout, 29.8 GiB of n-gram tables), same rig, same client, upstream's defaults (`--no-thinking
--context 8192`, six drafts):

| checkpoint | prose | code | prefill | loaded | start-up |
| --- | --- | --- | --- | --- | --- |
| ukisai NVFP4, `TENSORFOLD_FACES_FP8=1`, 16-pair items | 40.0 | 75.8 | 1486 | 97.39 GiB | 548 s |
| Vontra MLX-4bit | **50.1** | **97.7** | **2024** | 84.26 GiB | 271 s |

Its round is 49.2 ms against 73.6 ms on prose - and it accepts *fewer* drafts (25% against 31%), so the whole
difference is bytes a round reads, which is the point: the NVFP4 checkpoint's dense linears stay BF16. Prefill
separated by more (+69%) only against the mistuned item: with the 16-pair item it is 1486 against 2024 (+36%),
and the gap is the dense linears again - 4-bit throughout against BF16 rows re-read whole (prompts keep the
rows by design, so an 8-bit copy does not help a prompt either). The tables are not it: decode reads 0.6-0.8
KiB a token from them, and a 2275-token prompt pulls about 1.6 MB, a millisecond.

`tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --name swift-1.5 --host 0.0.0.0 --port 8083
--no-update-check --no-thinking --context 8192` serves it (the container needs the HF cache mounted and
`HF_HOME` pointing at it). Switch back to the NVFP4 checkpoint when byte-exact agreement with serial decoding
on that checkpoint's own format is what matters.

**45 tok/s on prose is the next 11%**, and what is left of it is host, not bytes. The round's device work is
~7.3 GB of routed experts plus the dense faces a round re-reads (6.9 GiB of stored BF16, 2.76 GB with the
8-bit copies above), and the projected profile shows what is left after them: 37.6 ms of kernels in a 59.6 ms
round, so ~15 ms a round is the host between graph replays - and the CPU profile names it. Per round the
forward gathers 72 n-gram rows through numpy's mmap and issues 22.6 pageable copies, and the thread pool that
serves them is where the round blocks; everything under 0.5 ms a launch amounts to 0.3 ms a round, so it is
not launch overhead and not small kernels. Read-ahead exists in the tree but for *prompt* chunks
(`families/qwen4_exp/runtime.py`, `engine/family_prefill.py`) and as a start-up page warm-up
(`engine.py`); the decode round's gather at `forward.py:244` is synchronous. Dispatching it for the round's
own ids before the verify is the change that closes prose, and it is a hot-loop one. The same model converted
4-bit throughout still leads (table above) - that gap is the checkpoint's layout, not the engine.

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
.env.sample             copy to .env: paths, port, HF token, EXTRA_ARGS
```

Weights, caches and logs stay on the host; the image holds the engine only. `scripts/build.sh` prints the
installed `tensorfold --version` and imports `tensorfold.families.qwen4_exp.cuda.nvfp4`, a module that exists
only on this branch, so a silently wrong build fails at build time.

## License

MIT (this repository). TensorFold is MIT; the model weights keep their own license on Hugging Face.
