#!/usr/bin/env bash
# Download the checkpoint into the host's Hugging Face cache (mounted at /hf in the container).
# Resumable: run it again after an interruption and snapshot_download finishes the missing files.
set -euo pipefail
cd "$(dirname "$0")/.."
# The caller's own settings win over .env: `TF_REF=<sha|branch> scripts/build.sh` and `IMAGE=...` are the
# knobs these scripts document, and sourcing .env with `set -a` after the caller's environment silently
# replaced them (a build asked for 1911880 came out as .env's pinned b4bf826, and the image it produced was
# the pinned one). The values that were in the environment are put back over .env's.
caller=()
for var in IMAGE MODEL HF_DIR; do
  [[ -n "${!var:-}" ]] && caller+=("${var}=${!var}")
done
[[ -f .env ]] && set -a && . ./.env && set +a
for entry in ${caller[@]+"${caller[@]}"}; do export "$entry"; done

IMAGE="${IMAGE:-tensorfold-spark:local}"
MODEL="${MODEL:-ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4}"
HF_DIR="${HF_DIR:-$HOME/.cache/tensorfold-spark/hf}"

mkdir -p "${HF_DIR}"
echo "[pull] ${MODEL} -> ${HF_DIR} (about 186 GB; the n-gram tables are most of it)"
echo "[pull] HF_TOKEN is passed through when this shell sets it (gated or rate-limited repos need it)"

docker run --rm --name tensorfold-spark-pull \
  -v "${HF_DIR}:/hf" \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN \
  "${IMAGE}" pull "${MODEL}"

echo "[pull] cached snapshot:"
find "${HF_DIR}/hub" -maxdepth 3 -name snapshot* -type d 2>/dev/null | head -2 || true
du -sh "${HF_DIR}" 2>/dev/null | tail -1 || true
