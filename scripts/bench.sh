#!/usr/bin/env bash
# Measure serving speed against a running server (prose, code, prefill). Run it while scripts/serve.sh runs.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a

IMAGE="${IMAGE:-swift-tensorfold:local}"
NAME="${NAME:-swift-1.5}"
PORT="${PORT:-8083}"
TOKENS="${TOKENS:-512}"

docker run --rm --network host \
  -v "$PWD/bench:/bench" -w /bench \
  -e MODEL="${NAME}" -e BASE="http://127.0.0.1:${PORT}" -e TOKENS="${TOKENS}" \
  --entrypoint python3 "${IMAGE}" \
  speed.py --base "http://127.0.0.1:${PORT}" --model "${NAME}" --tokens "${TOKENS}" \
  ${EXTRA_BENCH_ARGS:-} | tee bench/last-run.txt
