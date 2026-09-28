#!/usr/bin/env bash
# Check a running server: health, advertised model ids, then one timed chat completion.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a

PORT="${PORT:-8083}"
NAME="${NAME:-swift-1.5}"
BASE="http://127.0.0.1:${PORT}"

echo "== health =="
curl -fsS --max-time 10 "${BASE}/health"; echo

echo "== /v1/models =="
curl -fsS --max-time 10 "${BASE}/v1/models"; echo
echo "(--name ${NAME}: ask for that id, or use whatever /v1/models returns)"

echo "== chat completion =="
curl -fsS --max-time 300 -w '\ntotal %{time_total}s  http %{http_code}\n' \
  "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"En une phrase: que fait une table n-gram dans un modèle de langage ?\"}],\"max_tokens\":128}"

echo
echo "== expected =="
echo "health ok; the model id above; one short answer with usage and no error field."
