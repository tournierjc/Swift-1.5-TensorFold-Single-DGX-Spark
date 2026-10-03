#!/usr/bin/env bash
# Stop containers this repository started (serve is foreground by default; this covers detached runs and
# an interrupted download).
set -euo pipefail
for name in tensorfold-spark tensorfold-spark-pull; do
  if docker ps -q --filter "name=^${name}$" | grep -q .; then
    echo "[stop] ${name}"
    docker stop "${name}"
  else
    echo "[stop] ${name}: not running"
  fi
done
