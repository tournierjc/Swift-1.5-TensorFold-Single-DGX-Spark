#!/usr/bin/env bash
# Serve the checkpoint in the foreground, OpenAI-compatible endpoint on $PORT (default 8080).
# Ctrl-C stops it (or scripts/stop.sh when it runs detached).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a

IMAGE="${IMAGE:-swift-tensorfold:local}"
MODEL="${MODEL:-ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
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

exec docker run --rm --name swift-tensorfold \
  --gpus all --ipc=host --network host --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v "${HF_DIR}:/hf" -v "${STATE_DIR}:/state" -v "${MODELS_DIR}:/models:ro" \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN \
  "${IMAGE}" "${args[@]}"
