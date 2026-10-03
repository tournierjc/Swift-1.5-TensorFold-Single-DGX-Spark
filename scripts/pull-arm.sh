#!/usr/bin/env bash
# Download one arm's checkpoint into MODELS_DIR/<dir> and verify every file against the Hub's own
# sha256 (each entry's LFS oid), because nothing downstream re-reads those bytes.
#
# Why not `tensorfold pull`: it takes a repo id only. turboderp's EXL3 variants are *branches* of one
# repo, so a repo id resolves refs/main (metadata only, no weights) and the engine's cache resolver
# then falls back to "the newest config-bearing snapshot" (hub.cached) - i.e. whichever variant was
# downloaded last. A per-arm directory is unambiguous, and `serve` takes a directory path.
#
# curl -C - is used rather than the Hub client's xet path: measured on this host, one curl reaches
# 10.6 MB/s and four aggregate 8.7 MB/s, while an anonymous xet download managed ~1.7 MB/s.
#
#   scripts/pull-arm.sh exl3-405
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
ARMS="${ARMS:-bench/arms.json}"
arm="${1:?usage: pull-arm.sh <arm>}"

read -r repo rev dir <<<"$(python3 -c '
import json, sys
arm, path = sys.argv[1], sys.argv[2]
entry = json.load(open(path))["arms"].get(arm) or sys.exit(f"unknown arm {arm!r} in {path}")
print(entry["repo"], entry.get("revision") or "main", entry.get("dir") or entry["repo"].split("/")[-1])
' "$arm" "$ARMS")"

target="${MODELS_DIR}/${dir}"
base="https://huggingface.co/${repo}/resolve/${rev}"
mkdir -p "$target"
echo "[pull-arm] ${arm}: ${repo}@${rev}"
echo "[pull-arm] target ${target} (free: $(df -h "$target" | awk 'NR==2{print $4}'))"

python3 - "$repo" "$rev" >"$target/.manifest" <<'PY'
import json, sys, urllib.request

repo, rev = sys.argv[1], sys.argv[2]
url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
for entry in json.load(urllib.request.urlopen(url)):
    if entry.get("type") == "file":
        lfs = entry.get("lfs") or {}
        print(entry["path"], lfs.get("size") or entry.get("size") or 0, lfs.get("oid") or "-")
PY

while read -r path size oid; do
  if [ -s "$target/$path" ] && [ "$(stat -c%s "$target/$path")" = "$size" ]; then
    echo "[pull-arm] have ${path}"
    continue
  fi
  echo "[pull-arm] ${path} ($((size / 1000000)) MB) $(date -Is)"
  mkdir -p "$(dirname "$target/$path")"
  curl -sL -C - --retry 8 --retry-delay 5 -o "$target/$path" "$base/$path"
done <"$target/.manifest"

python3 - "$target" <<'PY'
import hashlib, pathlib, sys

target = pathlib.Path(sys.argv[1])
bad, checked = [], 0
for line in (target / ".manifest").read_text().splitlines():
    path, size, oid = line.rsplit(" ", 2)
    file = target / path
    if not file.is_file():
        bad.append(f"missing {path}")
        continue
    if file.stat().st_size != int(size):
        bad.append(f"size {path}: {file.stat().st_size} != {size}")
        continue
    if len(oid) == 64:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 22), b""):
                digest.update(chunk)
        if digest.hexdigest() != oid:
            bad.append(f"sha256 {path}")
            continue
    checked += 1
print(f"[pull-arm] verified {checked} files")
if bad:
    print("[pull-arm] FAILED:\n  " + "\n  ".join(bad[:20]))
    sys.exit(1)
PY
echo "[pull-arm] ${arm} complete: $(du -sh "$target" | cut -f1)"
