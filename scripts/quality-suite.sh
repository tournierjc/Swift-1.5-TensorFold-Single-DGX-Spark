#!/usr/bin/env bash
# Run the quality bench against the arm that is serving, and record what served it.
#
# The engine scores /v1/decisions from a single prefill: the prefix cache changes the *timing*, never the
# logits, so a replayed item scores identically and two arms can be compared item by item. What has to be
# controlled is the run itself, which is what this wrapper does: it refuses an output name that already
# exists, stamps the file with the arm and the minute, and captures the served arm's own startup line
# beside it - the allocated window, the room the plan left the streams' caches, whether the n-gram tables
# stayed resident - because two arms' numbers mean nothing without the profile they were produced under.
#
#   scripts/quality-suite.sh nvfp4-swift
#   scripts/quality-suite.sh exl3-405 --compare bench/quality/nvfp4-swift-20261003-031500.json
#
# Run the benchmark against a server started *for this run*: stop the previous arm, serve the new one,
# then run this. A missing --compare is the baseline case: the file it writes is what a later arm is
# compared against.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-http://127.0.0.1:8083}"
MODEL="${MODEL:-qwen3.8-flash-next}"
NAME="${NAME:-qwen3.8-flash-next}"
arm="${1:?usage: quality-suite.sh <arm> [quality.py options]}"
shift

stamp="$(date +%Y%m%d-%H%M%S)"
mkdir -p bench/quality
out="bench/quality/${arm}-${stamp}.json"
if [ -e "$out" ]; then
  echo "[quality-suite] $out already exists: a run is never reused, wait a minute or pick another arm name"
  exit 1
fi

# The arm's own provenance. Its absence is worth knowing too, so a failure here is not fatal.
docker logs "$NAME" 2>&1 | grep -E "startup estimate|Flash Next on CUDA|vision tower|mapped tables" \
  > "${out%.json}.startup.txt" || echo "[quality-suite] no startup lines from $NAME" > "${out%.json}.startup.txt"
curl -sf "$BASE/v1/models" > "${out%.json}.models.json" 2>/dev/null || echo "{}" > "${out%.json}.models.json"
echo "[quality-suite] serving arm lines:"
sed 's/^/  /' "${out%.json}.startup.txt"

python3 bench/quality.py --base "$BASE" --model "$MODEL" --arm "$arm" --json "$out" "$@"
echo "[quality-suite] run: $out"
echo "[quality-suite] provenance: ${out%.json}.startup.txt and ${out%.json}.models.json"
