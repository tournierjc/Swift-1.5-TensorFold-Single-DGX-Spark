#!/usr/bin/env bash
# Measure serving speed against a running server (prose, code, prefill). Run it while scripts/serve.sh runs.
set -euo pipefail
cd "$(dirname "$0")/.."
# The caller's own settings win over .env: `TF_REF=<sha|branch> scripts/build.sh` and `IMAGE=...` are the
# knobs these scripts document, and sourcing .env with `set -a` after the caller's environment silently
# replaced them (a build asked for 1911880 came out as .env's pinned b4bf826, and the image it produced was
# the pinned one). The values that were in the environment are put back over .env's.
caller=()
for var in IMAGE NAME PORT TOKENS EXTRA_BENCH_ARGS; do
  [[ -n "${!var:-}" ]] && caller+=("${var}=${!var}")
done
[[ -f .env ]] && set -a && . ./.env && set +a
for entry in ${caller[@]+"${caller[@]}"}; do export "$entry"; done

IMAGE="${IMAGE:-tensorfold-spark:local}"
NAME="${NAME:-qwen3.8-flash-next}"
PORT="${PORT:-8083}"
TOKENS="${TOKENS:-512}"

docker run --rm --network host \
  -v "$PWD/bench:/bench" -w /bench \
  -e MODEL="${NAME}" -e BASE="http://127.0.0.1:${PORT}" -e TOKENS="${TOKENS}" \
  --entrypoint python3 "${IMAGE}" \
  speed.py --base "http://127.0.0.1:${PORT}" --model "${NAME}" --tokens "${TOKENS}" \
  ${EXTRA_BENCH_ARGS:-} | tee bench/last-run.txt
