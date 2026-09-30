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
TF_REPO=https://github.com/tournierjc/TensorFold.git TF_REF=e125826475cdc6e62e35079f4dd501fa3f06c418 scripts/build.sh
```
