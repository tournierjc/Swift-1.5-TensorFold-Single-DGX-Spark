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
| Engine | `tournierjc/TensorFold@nvfp4-flash-next`, pinned to `491b2ad` (override with `TF_REF`) |
| Upstream PR | [ashhart/TensorFold#67](https://github.com/ashhart/TensorFold/pull/67) (draft) |
| Base image | `nvcr.io/nvidia/pytorch:26.07-py3` (36.5 GB as pulled here) — CUDA, torch 2.13, triton, the extension compiler |
| Endpoint | OpenAI-compatible on `:8080` (`/health`, `/v1/models`, `/v1/chat/completions`, streaming and tool calls) |

Pinned commit (in the `Dockerfile` as `ARG TF_REF`): `491b2ad837a30c6ef4815e587a65442f5f990f10`.

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
```

Logs go to the terminal that runs `scripts/serve.sh`; `scripts/stop.sh` stops a detached run and an
interrupted download.

## Serving on 128 GB

The first `serve` compiles kernels (triton and torch extensions) into `STATE_DIR`; later starts reuse them.
Then the levers, in `.env` as `EXTRA_ARGS`:

- `--ple-on-ssd` — the n-gram tables stay in the checkpoint on disk instead of the host page cache: about
  40 GiB less at peak for a few percent of decode speed. This is the first thing to try if the load is tight.
- `--ssd-experts 90` — stream routed experts into a 90 GiB GPU pool for models past the memory budget.
- `--mtp-drafts 6` — MTP drafts a round (the CUDA default; `0` disables drafting entirely).
- `--parallel 2` — two requests decoded together, windows sharing each round's forward (Flash Next, one rank).
- `--context N` — prompt plus reply window; the CUDA default is the affordable native capacity.
- `--no-drafts` — the serial reference: same output, slower.

Port 8080 here, 8083 in the vLLM rig — but only one of the two fits in memory at a time.

## Endpoint

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"swift-1.5","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128}'
```

## Status of the engine under test

Built and checked on the DGX Spark this rig targets:

- `scripts/build.sh` → image `swift-tensorfold:local`, 36.5 GB; `tensorfold --version` = 0.3.6.1 (from the
  pinned commit), and the build-time import of `families.qwen4_exp.cuda.nvfp4` — a module that exists only on
  this branch — passed, so a wrong ref fails at build time.
- Dependencies inside the image: hf-hub 1.24.0, tokenizers 0.23.1, safetensors 0.8.0, jinja2 3.1.6, numpy 2.1.0,
  torch 2.13.0a0 (NVIDIA's 26.07 container).
- `tensorfold info ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` inside the image resolved the family from the
  published `config.json`: `qwen4_exp`, 48 layers, 512 experts, `quantization modelopt`, "runs on NVIDIA GPUs
  (CUDA)".
- Hugging Face answers the repository listing anonymously from the Spark (100 files, 31 `embedding-model-*`),
  so `scripts/pull.sh` needs no token today; keep `HF_TOKEN` in `.env` for the day that changes.

The branch's CUDA suite runs green in the same container this image is built from: **460 passed, 70 skipped,
0 failed** on a GB10 (upstream `main` 0.3.6.1 in that container: 429 passed, 70 skipped). Against the real
published weights, checked in-process:

- 16 routed experts bit-exact against the reference dequantization; FP4 matmul error 2.6e-3 / 2.3e-3; BF16
  faces 2.9e-3; row invariance from 1 to 64 rows; 512 MTP experts exact; the n-gram hashing constants equal
  value for value.
- The n-gram table on real bytes: shard 2 of `embedding-model-00006-of-00131.safetensors` gathers rows
  bit-identical to an independent read of the file, and 128 shards of that size make the 320,001,536-row
  table the digest derives.

Not verified yet, and worth reporting from a run here:

- **A full load and a decode round on the complete 186 GB checkpoint.** The work to date loaded embed, mixer
  and layer 0 on real shards, then stopped at layer 1's PLE when that table was the blocker; the table is what
  this rig is meant to exercise end to end.
- CUDA-graph capture with the FP4 MoE route: the MoE step walks a host-side item list, so a captured graph
  encodes one step's list. Try `--parallel` off first, then with graphs enabled.
- Throughput and memory peak on the Spark, and the effect of `--ple-on-ssd`.
- One of the 128 n-gram shards is proven against real bytes; the other 127 are read by the same code path and
  each shard's header is checked at load time, a mixed layout raising rather than loading wrongly.

## Troubleshooting

- **`KeyError: ...ngram_embedding.shard_0.weight`** — the checkpoint is incomplete. A repacked copy (48 layer
  files, no `embedding-*` files ~79 GB, as some local copies are) has no n-gram table: download the published
  revision with `scripts/pull.sh` and serve that.
- **`401`/`403` from Hugging Face** — set `HF_TOKEN` in `.env` (a read token); the scripts pass it through and
  never write it anywhere else.
- **`mlock` warnings on start** — the PLE tables are pinned when the memory-lock limit allows it; the run
  continues without the pin. `scripts/serve.sh` already raises the limit and adds `IPC_LOCK`.
- **Slow first token after a rebuild** — the kernels are JIT-compiled on first use; keep `STATE_DIR` across
  rebuilds.
- **Out of memory at load** — add `--ple-on-ssd`, then `--ssd-experts`, then lower `--context`.

## Layout

```
Dockerfile              image: NVIDIA PyTorch + TensorFold (pinned ref), weights never baked in
scripts/build.sh        docker build, then `tensorfold --version` and a branch-only import check
scripts/pull.sh         resumable download into HF_DIR
scripts/serve.sh        foreground serve with the Spark's mounts, caps and flags
scripts/smoke.sh        health, model ids, one timed completion
scripts/stop.sh         stop this rig's containers
.env.sample             copy to .env: paths, port, HF token, EXTRA_ARGS
```

Weights, caches and logs stay on the host; the image holds the engine only. `scripts/build.sh` prints the
installed `tensorfold --version` and imports `tensorfold.families.qwen4_exp.cuda.nvfp4`, a module that exists
only on this branch, so a silently wrong build fails at build time.

## License

MIT (this repository). TensorFold is MIT; the model weights keep their own license on Hugging Face.
