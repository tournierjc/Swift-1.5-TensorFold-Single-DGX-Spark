#!/usr/bin/env bash
# Convert an EXL3 arm's quantized vision sidecar into the floating tower the CUDA engine loads.
#
# An EXL3 pack keeps its vision tower outside the model index (turboderp's packs ship
# vision_k6.safetensors, ~0.56 GB of trellis-coded tensors), and the loader reads a *floating* tower:
# tensorfold.vision.exl3_convert dequantizes it, rebuilds the fused QKV the ExLlamaV3 layout splits,
# strips the MLP padding and writes an F16 artifact that records its source sha256, codec and converter
# version. One-time and idempotent - re-running against the same source returns the same artifact unless
# the source changed, in which case it refuses rather than overwrite.
#
# The output lives beside the sidecar, so `MODEL=/models/<dir>` and
# TENSORFOLD_VISION_WEIGHTS=/models/<dir>/vision-f16.safetensors are the two settings a vision serve needs
# (scripts/serve.sh forwards the variable; integration/0.6.3 + d31685e names this command when it is unset).
#
#   scripts/convert-vision.sh exl3-405
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-swift-tensorfold:local}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
ARMS="${ARMS:-bench/arms.json}"
arm="${1:?usage: convert-vision.sh <arm>}"

dir="$(python3 -c '
import json, sys
entry = json.load(open(sys.argv[2]))["arms"].get(sys.argv[1]) or sys.exit(f"unknown arm {sys.argv[1]!r}")
print(entry.get("dir") or entry["repo"].split("/")[-1])
' "$arm" "$ARMS")"
target="${MODELS_DIR}/${dir}"

sidecar="$(ls "$target"/vision_k*.safetensors 2>/dev/null | head -1 || true)"
if [ -z "$sidecar" ]; then
  echo "[convert-vision] ${arm}: no vision_k*.safetensors in ${target} - nothing to convert"
  echo "[convert-vision] (a ModelOpt/NVFP4 pack carries a floating tower inside its shards; leave this alone)"
  exit 0
fi

# -u so the artifact is owned by the caller: a root-owned file in MODELS_DIR then blocks its own cleanup.
docker run --rm -u "$(id -u):$(id -g)" -v "${target}:/arm" -w /arm --entrypoint python3 "${IMAGE}" \
  -m tensorfold.vision.exl3_convert "/arm/$(basename "$sidecar")" /arm/vision-f16.safetensors
ls -la "$target/vision-f16.safetensors"
echo "[convert-vision] serve with TENSORFOLD_VISION_WEIGHTS=/models/${dir}/vision-f16.safetensors"
