#!/usr/bin/env bash
# Build the test image on the Spark. Local only: nothing is pushed to a registry.
set -euo pipefail
cd "$(dirname "$0")/.."
# The caller's own settings win over .env: `TF_REF=<sha|branch> scripts/build.sh` and `IMAGE=...` are the
# knobs these scripts document, and sourcing .env with `set -a` after the caller's environment silently
# replaced them (a build asked for 1911880 came out as .env's pinned b4bf826, and the image it produced was
# the pinned one). The values that were in the environment are put back over .env's.
caller=()
for var in TF_REPO TF_REF IMAGE; do
  [[ -n "${!var:-}" ]] && caller+=("${var}=${!var}")
done
[[ -f .env ]] && set -a && . ./.env && set +a
for entry in ${caller[@]+"${caller[@]}"}; do export "$entry"; done

IMAGE="${IMAGE:-swift-tensorfold:local}"
args=()
[[ -n "${TF_REF:-}" ]] && args+=(--build-arg "TF_REF=${TF_REF}")
# TF_REPO too: the Dockerfile's default is the upstream, so without this a fork branch can never be
# built whatever ref is asked for - pip reports it as `pathspec ... did not match any file(s)`.
[[ -n "${TF_REPO:-}" ]] && args+=(--build-arg "TF_REPO=${TF_REPO}")

echo "[build] image=${IMAGE}${TF_REF:+  tensorfold ref=${TF_REF}}"
docker build "${args[@]}" -t "${IMAGE}" .

echo "[build] installed package:"
docker run --rm "${IMAGE}" --version
docker run --rm --entrypoint python3 "${IMAGE}" -c \
  "from tensorfold.families.qwen4_exp.cuda import nvfp4, nvfp4_moe; print('NVFP4 route present (branch-only module)')"
echo "[build] done: ${IMAGE}"
