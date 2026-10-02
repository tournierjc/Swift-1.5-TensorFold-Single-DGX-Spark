#!/usr/bin/env bash
# The whole bench suite against a running server, as one arm of an A/B. Run it on the Spark host while
# scripts/serve.sh runs, once per arm, so the two logs compare line for line.
#
#   scripts/bench-suite.sh before-061            # writes bench/before-061.log, bench/before-061-speed.json
#   IMAGE=swift-tensorfold:local scripts/bench-suite.sh after-062
#
# What it runs, in this order, and why this order:
#   a discard pass  -- the rig's rule: the first pass after a load is cold, and an undeclared transient
#                      masquerades as the effect of whatever was just changed;
#   aggregate 1/2/3 -- bench/aggregate.py, distinct prompts, 1024-token replies: a round is weight-bound and
#                      the lanes share its reads, so the aggregate is the number that shows it;
#   speed           -- bench/speed.py, prose/code/prefill, 512-token replies, two rounds each;
#   vision          -- bench/vision_probe.py, which generates what it asks about and takes the colour as an
#                      argument, so the answer has to come from the pixels.
# It records which image was *serving* (the container's own image id, not the tag you meant to run), because
# two revisions of one engine print the same startup lines and the tag is not evidence.
set -uo pipefail
cd "$(dirname "$0")/.."
caller=()
for var in IMAGE NAME PORT ARM; do
  [[ -n "${!var:-}" ]] && caller+=("${var}=${!var}")
done
[[ -f .env ]] && set -a && . ./.env && set +a
for entry in ${caller[@]+"${caller[@]}"}; do export "$entry"; done

IMAGE="${IMAGE:-swift-tensorfold:local}"
NAME="${NAME:-swift-1.5}"
PORT="${PORT:-8083}"
ARM="${ARM:-${1:-arm-$(date +%Y%m%d-%H%M%S)}}"
BASE="http://127.0.0.1:${PORT}"
LOG="bench/${ARM}.log"
mkdir -p bench
exec > >(tee "$LOG") 2>&1

echo "=== bench suite: ${ARM}  (bench image ${IMAGE})  started $(date -Is) ==="
echo "--- what is serving ---"
docker ps --format '{{.Names}} | {{.Image}} | {{.Status}}' | grep -E 'swift|tensorfold' || true
served="$(docker ps --format '{{.Names}}' | grep -m1 tensorfold || true)"
[[ -n "${served}" ]] && echo "served container: ${served}  image id $(docker inspect -f '{{.Image}}' "${served}")"
# The serve's own startup lines, if its log is around: the allocated window and the vision reserve are the two
# settings whose *effect* cannot be read from the flags that were passed.
grep -hE "startup estimate|vision:|Flash Next on CUDA|loaded in" "${HOME}"/serve-*.log 2>/dev/null | tail -4
echo "--- health ---"
curl -fsS -m 10 "${BASE}/health"; echo; echo

echo "--- discard pass (the first pass after a load is cold) ---"
time curl -fsS -m 900 "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}],\"max_tokens\":64}" \
  > /dev/null
echo

echo "--- aggregate, 1/2/3 clients, 1024-token replies, distinct prompts ---"
docker run --rm --network host -v "$PWD/bench:/bench" --entrypoint python3 "${IMAGE}" \
  /bench/aggregate.py --base "${BASE}" --model "${NAME}" --tokens 1024 --clients 1 2 3
echo

echo "--- client-side speed, 512-token replies, 2 rounds ---"
docker run --rm --network host -v "$PWD/bench:/bench" -w /bench --entrypoint python3 "${IMAGE}" \
  speed.py --base "${BASE}" --model "${NAME}" --tokens 512 --rounds 2 --json "/bench/${ARM}-speed.json"
echo

echo "--- vision: red image and red clip ---"
docker run --rm --network host -v "$PWD/bench:/bench" -w /bench --entrypoint python3 "${IMAGE}" \
  /bench/vision_probe.py --base "${BASE}" --model "${NAME}" --colour red --cases image video
echo

echo "--- vision: blue image ---"
docker run --rm --network host -v "$PWD/bench:/bench" -w /bench --entrypoint python3 "${IMAGE}" \
  /bench/vision_probe.py --base "${BASE}" --model "${NAME}" --colour blue --cases image
echo

echo "--- server counters after the suite ---"
curl -fsS -m 10 "${BASE}/health"; echo
echo "=== bench suite done: ${ARM}  finished $(date -Is) ==="
