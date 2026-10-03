#!/usr/bin/env bash
# Serve the checkpoint in the foreground, OpenAI-compatible endpoint on $PORT (default 8083).
# Ctrl-C stops it (or scripts/stop.sh when it runs detached).
set -euo pipefail
cd "$(dirname "$0")/.."
# The caller's own settings win over .env: `TF_REF=<sha|branch> scripts/build.sh` and `IMAGE=...` are the
# knobs these scripts document, and sourcing .env with `set -a` after the caller's environment silently
# replaced them (a build asked for 1911880 came out as .env's pinned b4bf826, and the image it produced was
# the pinned one). The values that were in the environment are put back over .env's.
caller=()
# Every knob docker_env forwards has to be listed here too: one that is not comes back from .env, and the
# arm you asked for is not the arm you measured (TENSORFOLD_FACES_FP8=xall served .env's all, twice).
for var in IMAGE MODEL NAME HOST PORT CONTEXT PARALLEL EXTRA_ARGS \
           HF_DIR STATE_DIR MODELS_DIR \
           TORCH_USE_CUDA_DSA CUDA_LAUNCH_BLOCKING PYTORCH_CUDA_ALLOC_CONF \
           TENSORFOLD_SKIP_WARM TENSORFOLD_NVFP4_MOE TENSORFOLD_FACES_FP8 \
           TENSORFOLD_EXL3_MIDM TENSORFOLD_EXL3_WC TENSORFOLD_EXL3_FOLD TENSORFOLD_EXL3_FOLD_BF16 TENSORFOLD_EXL3_FOLD2 \
           TENSORFOLD_VISION_WORKSPACE_MIB TENSORFOLD_VISION_WEIGHTS; do
  [[ -n "${!var:-}" ]] && caller+=("${var}=${!var}")
done
# The arm flag picks the pack: `scripts/serve.sh --exl3-405-turboderp`, `--nvfp4-swift`, `--nvfp4-radixart`,
# `--nvfp4-nvidia`, `--nvfp4-local-inference-lab`, `--exl3-605-turboderp`, `--exl3-305-turboderp` - the arms of
# `bench/arms.json`. It selects the model, the state directory and the runtime flags that arm needs, so two arms of
# one engine revision cannot be confused in a deploy line or a result file. Without a flag, .env decides, as
# before. `--exl3` still means the 4.05bpw pack; `--nvfp4` alone is refused rather than guessed, because three
# packs answer to it now.
ARM=""
for arg in "$@"; do
  case "$arg" in
    --exl3)  ARM="exl3-405-turboderp"; echo "[serve] --exl3 is now --exl3-405-turboderp" ;;
    --nvfp4) echo "[serve] --nvfp4 names no pack: use --nvfp4-swift, --nvfp4-radixart, --nvfp4-nvidia or --nvfp4-local-inference-lab" >&2; exit 2 ;;
    --*=*)   : ;;                      # a --flag=value belongs to the engine, not to this script
    --*)     ARM="${arg#--}" ;;
  esac
done
[[ -f .env ]] && set -a && . ./.env && set +a
for entry in ${caller[@]+"${caller[@]}"}; do export "$entry"; done

# Everything this rig owns lives under one cache root named after the rig, not after a checkpoint: the state
# directories of the arms sit beside the HF cache instead of under `~/.cache/swift-tensorfold`.
RIG_CACHE="${RIG_CACHE:-$HOME/.cache/tensorfold-spark}"
case "${ARM:-}" in
  exl3-405-turboderp)        MODEL="/models/exl3-405"; STATE_DIR="$RIG_CACHE/state-exl3-405-turboderp" ;;
  exl3-605-turboderp)        MODEL="/models/exl3-605"; STATE_DIR="$RIG_CACHE/state-exl3-605-turboderp" ;;
  exl3-305-turboderp)        MODEL="/models/exl3-305"; STATE_DIR="$RIG_CACHE/state-exl3-305-turboderp" ;;
  nvfp4-swift)               MODEL="ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4";    STATE_DIR="$RIG_CACHE/state-nvfp4-swift" ;;
  nvfp4-radixart)            MODEL="RadixArk/Qwen3.8-Flash-Next-NVFP4";            STATE_DIR="$RIG_CACHE/state-nvfp4-radixart" ;;
  nvfp4-nvidia)              MODEL="nvidia/Qwen3.8-Flash-Next-NVFP4";              STATE_DIR="$RIG_CACHE/state-nvfp4-nvidia" ;;
  nvfp4-local-inference-lab) MODEL="local-inference-lab/Qwen3.8-Flash-Next-NVFP4"; STATE_DIR="$RIG_CACHE/state-nvfp4-local-inference-lab" ;;
  "") : ;;
  *)  echo "[serve] unknown arm '${ARM}': bench/arms.json lists them (--exl3-405-turboderp, --nvfp4-swift, --nvfp4-radixart, --nvfp4-nvidia, ...)" >&2; exit 2 ;;
esac
# The tag is the caller's, not this script's: it is what a result file will say, so it must name the arm that
# really served. Rewriting it silently hid exactly that (an image whose tag said `nvfp4` served the EXL3 pack).
if [[ -n "${ARM:-}" && "${IMAGE:-}" == tensorfold-spark:* ]]; then
  case "${IMAGE#tensorfold-spark:}" in
    "${ARM}"|"${ARM}"-*) : ;;
    *) echo "[serve] warning: IMAGE=${IMAGE} does not name the arm being served (${ARM})" >&2 ;;
  esac
fi

IMAGE="${IMAGE:-tensorfold-spark:local}"
MODEL="${MODEL:-ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8083}"
NAME="${NAME:-qwen3.8-flash-next}"
# The window is prompt plus reply, and unset is the CUDA's own affordable native capacity - what a server that
# people talk to wants, since a long conversation needs room and only prefill feels the difference. The benches
# in the README pin CONTEXT=8192 to match the numbers they quote, but a window is not a rate: the same
# 2275-token prompt measures 1509-1533 tok/s at the full native 262144 and at 8192 alike (three runs each), and
# the full native window costs about 1.2 GiB more than an 8k one. Unset is what a server people talk to wants.
CONTEXT="${CONTEXT:-}"
# PARALLEL is worth setting: on CUDA the server's own default is one request decoded at a time, the others
# queued, so N sessions cost N times the wall time. An explicit number batches them - four concurrent clients
# measured 99.4 tok/s aggregate against 30.7 for one, at 8.4 s against 6.8 s each - because a round is
# weight-bound and the lanes share its reads. Lanes share the slot pool, so N lanes of --context C want N x C
# slots at about 4.8 KB each. With more than one lane the window may also be *silently* clamped instead of
# refused - `--parallel 3 --context 262144` was accepted and then allocated 8192 - so read the server's
# "allocated prompt/reply window" line in the log after starting, not the flags you passed.
PARALLEL="${PARALLEL:-}"
HF_DIR="${HF_DIR:-$RIG_CACHE/hf}"
STATE_DIR="${STATE_DIR:-$RIG_CACHE/state}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

mkdir -p "${HF_DIR}" "${STATE_DIR}" "${MODELS_DIR}"

args=(serve "${MODEL}" --host "${HOST}" --port "${PORT}" --name "${NAME}"
      --snapshot-dir /state/snapshots --no-update-check)
[[ -n "${CONTEXT}" ]] && args+=(--context "${CONTEXT}")
[[ -n "${PARALLEL}" ]] && args+=(--parallel "${PARALLEL}")
# shellcheck disable=SC2206  # EXTRA_ARGS is deliberately word-split: it holds serve flags
[[ -n "${EXTRA_ARGS:-}" ]] && args+=(${EXTRA_ARGS})

echo "[serve] ${MODEL}  ->  http://${HOST}:${PORT}/v1   (model id: ${NAME})"
echo "[serve] flags: ${args[*]}"
echo "[serve] the first start compiles kernels into ${STATE_DIR}; the 186 GB snapshot comes from ${HF_DIR}"

# The CUDA crash switches, forwarded only when set. TORCH_USE_CUDA_DSA=1 turns a bad access into a device-side
# assert naming the kernel and line; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True helps fragmentation.
# CUDA_LAUNCH_BLOCKING=1 serialises launches, so keep it off while graphs are captured.
# TENSORFOLD_FACES_FP8=1 loads the DeltaNet and attention linears with an 8-bit copy beside the stored BF16
# rows (all: every BF16 face), which a round reads instead - see "8-bit dense faces" in the README. It is off
# unless set: the copy is coarser than the rows, and the MTP head drafts less well against a body it was not
# calibrated for, so measure it before keeping it.
# The EXL3 kernel switches, read in the engine at import. `perf/exl3-midm-wc` takes the two kernels that
# stop `linear_kernel` from decoding every tile once per 16-row pass: TENSORFOLD_EXL3_MIDM=0 keeps
# `linear_kernel` at every row count, TENSORFOLD_EXL3_WC=0 leaves 17-128 rows to those mid-M kernels
# instead of `linear_wc`. Both only bite from 17 rows up, which on this rig means three streams' verify
# windows batched together - one stream is 7 rows and sees the old kernel either way. TENSORFOLD_EXL3_FOLD=0
# puts the prompt back on the decode-rotation W_q path instead of the once-a-call folded W'' = diag(suh) H
# W_q H / 128; FOLD_BF16 and FOLD2 pick that path's dtype and decoder. Those two kernels are the rig's port
# of the *open* upstream PR ashhart/TensorFold#260 (same bits per commit), so a pin that loses them turns
# both switches into silent no-ops: the build imports their reader for that reason.
# TENSORFOLD_EXL3_FDIRECT_ROWS is **not** forwarded: the flag has no reader in the served revision. The
# fdirect prompt path it set (a short 4-bit call rebuilding W'' inside the GEMM) left the branch with the
# layer-major prompt rewrite (`9b13c50`, the shared half of the 27b prompt commit), and `prefill.py` no longer
# carries the constant - only the kernel in `cuda/exl3/linear.cu`, unbound. The three stale references are the
# engine's `docs/recipes/exl3.md`, the test in `tests/cuda/test_exl3_prefill.py` that monkeypatches the constant
# (it has nothing to set now) and this script's old comment. They are a defect on the *old* revision too, not
# something the 0.6.4 rebase introduced; the fork-side doc and test are left for a unit of their own so the
# rebase stays a pure rebase.
# TENSORFOLD_VISION_WORKSPACE_MIB is the tower's workspace *reserve*. The engine's own default is 4 GiB (0.6.1
# and 0.6.2 alike) whatever the tower's own estimate says, and the reserve is counted against the startup
# admission, so an unset value plans ~2.7 GiB more than this rig measured with. 1280 is what this rig sets
# (measured peaks: 0.76 GiB an image, 0.83 GiB a video).
# TENSORFOLD_VISION_WEIGHTS is the one variable the serving environment cannot infer: an EXL3 pack keeps its
# vision tower in a quantized sidecar the model index does not list (turboderp's packs ship vision_k6.safetensors),
# and the loader reads a *floating* tower. scripts/convert-vision.sh converts the sidecar once with the engine's
# own tensorfold.vision.exl3_convert, and this variable points at that artifact; unset, an EXL3 arm with --vision
# refuses to start (the engine names the converter since integration/0.6.3 + d31685e).
docker_env=()
for var in TORCH_USE_CUDA_DSA CUDA_LAUNCH_BLOCKING PYTORCH_CUDA_ALLOC_CONF TENSORFOLD_SKIP_WARM TENSORFOLD_NVFP4_MOE TENSORFOLD_FACES_FP8 TENSORFOLD_EXL3_MIDM TENSORFOLD_EXL3_WC TENSORFOLD_EXL3_FOLD TENSORFOLD_EXL3_FOLD_BF16 TENSORFOLD_EXL3_FOLD2 TENSORFOLD_VISION_WORKSPACE_MIB TENSORFOLD_VISION_WEIGHTS; do
  [[ -n "${!var:-}" ]] && docker_env+=(-e "${var}")
done

# A container of this name already running is either a live server or the corpse of a crashed one. Replace it:
# otherwise docker refuses the run, serve.sh fails, and the caller's smoke test talks to the old process.
docker rm -f tensorfold-spark >/dev/null 2>&1 || true

exec docker run --rm --name tensorfold-spark \
  --gpus all --ipc=host --network host --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v "${HF_DIR}:/hf" -v "${STATE_DIR}:/state" -v "${MODELS_DIR}:/models:ro" \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN "${docker_env[@]}" \
  "${IMAGE}" "${args[@]}"
