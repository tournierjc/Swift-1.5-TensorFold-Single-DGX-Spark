#!/usr/bin/env bash
# Download the checkpoint into the host's Hugging Face cache (mounted at /hf in the container).
# Resumable: run it again after an interruption and snapshot_download finishes the missing files.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a

IMAGE="${IMAGE:-swift-tensorfold:local}"
MODEL="${MODEL:-ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4}"
HF_DIR="${HF_DIR:-$HOME/.cache/swift-tensorfold/hf}"

mkdir -p "${HF_DIR}"
echo "[pull] ${MODEL} -> ${HF_DIR} (about 186 GB; the n-gram tables are most of it)"
echo "[pull] HF_TOKEN is passed through when this shell sets it (gated or rate-limited repos need it)"

docker run --rm --name swift-tensorfold-pull \
  -v "${HF_DIR}:/hf" \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN \
  "${IMAGE}" pull "${MODEL}"

echo "[pull] cached snapshot:"
find "${HF_DIR}/hub" -maxdepth 3 -name snapshot* -type d 2>/dev/null | head -2 || true
du -sh "${HF_DIR}" 2>/dev/null | tail -1 || true
