#!/usr/bin/env bash
# Build the test image on the Spark. Local only: nothing is pushed to a registry.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a

IMAGE="${IMAGE:-swift-tensorfold:local}"
args=()
[[ -n "${TF_REF:-}" ]] && args+=(--build-arg "TF_REF=${TF_REF}")

echo "[build] image=${IMAGE}${TF_REF:+  tensorfold ref=${TF_REF}}"
docker build "${args[@]}" -t "${IMAGE}" .

echo "[build] installed package:"
docker run --rm "${IMAGE}" --version
docker run --rm --entrypoint python3 "${IMAGE}" -c \
  "from tensorfold.families.qwen4_exp.cuda import nvfp4, nvfp4_moe; print('NVFP4 route present (branch-only module)')"
echo "[build] done: ${IMAGE}"
