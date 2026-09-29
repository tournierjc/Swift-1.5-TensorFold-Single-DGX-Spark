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
for var in IMAGE MODEL NAME HOST PORT EXTRA_ARGS; do
  [[ -n "${!var:-}" ]] && caller+=("${var}=${!var}")
done
[[ -f .env ]] && set -a && . ./.env && set +a
for entry in ${caller[@]+"${caller[@]}"}; do export "$entry"; done

IMAGE="${IMAGE:-swift-tensorfold:local}"
MODEL="${MODEL:-ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8083}"
NAME="${NAME:-swift-1.5}"
HF_DIR="${HF_DIR:-$HOME/.cache/swift-tensorfold/hf}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/swift-tensorfold/state}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

mkdir -p "${HF_DIR}" "${STATE_DIR}" "${MODELS_DIR}"

args=(serve "${MODEL}" --host "${HOST}" --port "${PORT}" --name "${NAME}"
      --snapshot-dir /state/snapshots --no-update-check)
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
docker_env=()
for var in TORCH_USE_CUDA_DSA CUDA_LAUNCH_BLOCKING PYTORCH_CUDA_ALLOC_CONF TENSORFOLD_SKIP_WARM TENSORFOLD_NVFP4_MOE TENSORFOLD_FACES_FP8; do
  [[ -n "${!var:-}" ]] && docker_env+=(-e "${var}")
done

# A container of this name already running is either a live server or the corpse of a crashed one. Replace it:
# otherwise docker refuses the run, serve.sh fails, and the caller's smoke test talks to the old process.
docker rm -f swift-tensorfold >/dev/null 2>&1 || true

exec docker run --rm --name swift-tensorfold \
  --gpus all --ipc=host --network host --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v "${HF_DIR}:/hf" -v "${STATE_DIR}:/state" -v "${MODELS_DIR}:/models:ro" \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN "${docker_env[@]}" \
  "${IMAGE}" "${args[@]}"
