#!/usr/bin/env bash
# Size one arm's serving plan before loading it: the engine's own admission, header-only, in seconds.
#
# Wraps scripts/preflight-arm.py in the engine image with the arm mounted read-only, so the receipt comes
# from the same code path that will serve it. On this host the plan is admitted against MemAvailable, so
# nothing may be holding the device: this is a *stops-the-server* check, not a side-by-side one.
#
#   scripts/preflight-arm.sh exl3-405
#   scripts/preflight-arm.sh exl3-405 --streams 4 --context 262144 --vision --budget-gib 104.76
#   scripts/preflight-arm.sh /home/jct-spark/probe/stub-405 --streams 4 --vision --budget-gib 104.76
#
# --budget-gib plans against a number instead of the live MemAvailable: the way to ask "would this fit
# once the server is down" without stopping it. Exit status is 1 if any stream count is refused or if the
# prompt/reply window comes back clamped below --context (a clamp is silent in the engine's own log line,
# and it has happened on this rig: --parallel 3 --context 262144 came back as 8192).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-swift-tensorfold:local}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
ARMS="${ARMS:-bench/arms.json}"
arm="${1:?usage: preflight-arm.sh <arm|path> [preflight-arm.py options]}"
shift

if [ -d "$arm" ]; then
  target="$(cd "$arm" && pwd)"
  dir="$(basename "$target")"
  if [ "$(dirname "$target")" != "$(cd "$MODELS_DIR" && pwd)" ]; then
    MODELS_DIR="$(dirname "$target")"
  fi
else
  dir="$(python3 -c '
import json, sys
entry = json.load(open(sys.argv[2]))["arms"].get(sys.argv[1]) or sys.exit(f"unknown arm {sys.argv[1]!r}")
print(entry.get("dir") or entry["repo"].split("/")[-1])
' "$arm" "$ARMS")"
  target="${MODELS_DIR}/${dir}"
fi
[ -d "$target" ] || { echo "[preflight-arm] no such directory: $target"; exit 1; }

docker_env=(-e "PYTHONDONTWRITEBYTECODE=1")
if [ -f "$target/vision-f16.safetensors" ]; then
  docker_env+=(-e "TENSORFOLD_VISION_WEIGHTS=/models/${dir}/vision-f16.safetensors")
fi
for var in TENSORFOLD_SKIP_WARM TENSORFOLD_NVFP4_MOE TENSORFOLD_FACES_FP8 TENSORFOLD_VISION_WORKSPACE_MIB TENSORFOLD_VISION_WEIGHTS; do
  [[ -n "${!var:-}" ]] && docker_env+=(-e "${var}=${!var}")
done

if docker ps --format '{{.Names}}' | grep -q '^swift-tensorfold$'; then
  echo "[preflight-arm] WARNING: swift-tensorfold is up and holds the device: the budget below is not idle"
fi

# Sizing an arm before its bytes are on disk means a stub directory: files of the right size with their
# contents elsewhere, linked to the real ones. Those links have to resolve inside the container, and they
# are absolute host paths, so the targets are mounted at the same path:
#   EXTRA_MOUNTS=/home/jct-spark/models:/home/jct-spark/models:ro scripts/preflight-arm.sh ~/probe/stub-405 ...
extra=()
for mount in ${EXTRA_MOUNTS:-}; do
  [[ -n "$mount" ]] && extra+=(-v "$mount")
done

docker run --rm --gpus all "${docker_env[@]}" "${extra[@]}" \
  -v "${MODELS_DIR}:/models:ro" -v "${PWD}:/recipe:ro" \
  --entrypoint python3 "${IMAGE}" /recipe/scripts/preflight-arm.py "/models/${dir}" "$@"
