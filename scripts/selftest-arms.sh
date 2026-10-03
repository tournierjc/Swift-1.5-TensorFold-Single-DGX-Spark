#!/usr/bin/env bash
# Assert the arm flag's contract without a GPU, without a container, and without touching this rig's own .env.
#
# The launcher is dispatched once per arm from a scratch copy of the tree, with a `docker` stub on PATH that
# prints both the run line it was handed and the engine-visible environment the container would get. What an arm
# selects is then readable, and a regression in the wiring fails here instead of surfacing as an arm that served
# as something else - which is the defect this file exists for: the registry has recorded `flags` for every arm
# since it was written, and nothing applied them, so `--nvfp4-swift` on this rig's `.env` (written for the EXL3
# arm) used to serve the NVFP4 arm without `TENSORFOLD_FACES_FP8=all`, the largest lever that arm has, in silence.
#
#   scripts/selftest-arms.sh
set -euo pipefail
cd "$(dirname "$0")/.."

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/tree/scripts" "$tmp/tree/bench"
cp scripts/serve.sh "$tmp/tree/scripts/"
cp bench/arms.json "$tmp/tree/bench/"

cat > "$tmp/bin/docker" <<'STUB'
#!/usr/bin/env bash
echo "DOCKER $*"
env | grep -E '^(TENSORFOLD_[A-Z0-9_]+|PARALLEL|MODEL|IMAGE)=' | sort
STUB
chmod +x "$tmp/bin/docker"

# A `.env` written for another arm, which is the situation the rig is always in: it carries the EXL3 switches and
# a FACES_FP8 the NVFP4 arms must not inherit. Paths are the scratch tree's, so the test touches nothing.
cat > "$tmp/tree/.env" <<ENV
IMAGE=tensorfold-spark:selftest
MODEL=/models/from-env
HF_DIR=$tmp/cache/hf
STATE_DIR=$tmp/cache/state-from-env
MODELS_DIR=$tmp/models
TENSORFOLD_EXL3_MIDM=1
TENSORFOLD_FACES_FP8=1
ENV

fails=0
declare -a out
run() { # run [VAR=value ...] <arm args...>
  local -a envs=()
  while (( $# )) && [[ "$1" == *=* ]]; do envs+=("$1"); shift; done
  out=$( cd "$tmp/tree" && env "${envs[@]}" PATH="$tmp/bin:$PATH" bash scripts/serve.sh "$@" 2>&1 ) && rc=0 || rc=$?
}
ok()   { echo "  ok    $1"; }
bad()  { echo "  FAIL  $1"; fails=$((fails + 1)); }
has()     { grep -q -- "$2" <<< "$1" && ok "$3" || bad "$3 (missing: $2)"; }
hasnt()   { grep -q -- "$2" <<< "$1" && bad "$3 (unexpected: $2)" || ok "$3"; }
refuses() { [[ "${rc:-0}" -ne 0 ]] && has "$1" "$2" "$3" || bad "$3 (exit $rc: $2)"; }

echo "1. an arm's own flags come from the registry and beat a .env written for another arm"
run --nvfp4-swift
has "$out" '\[serve\] nvfp4-swift sets TENSORFOLD_FACES_FP8=all' "the registry's value is applied and said out loud"
has "$out" '^TENSORFOLD_FACES_FP8=all$' "the container sees all, not the .env's 1"
has "$out" '\-e TENSORFOLD_FACES_FP8' "the variable is forwarded"
has "$out" "state-nvfp4-swift:/state" "the arm's own state directory is mounted"
has "$out" 'ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4' "the arm's model is served"

echo "2. the caller's own environment still wins over the registry"
run TENSORFOLD_FACES_FP8=xall --nvfp4-swift
has "$out" '^TENSORFOLD_FACES_FP8=xall$' "an exported value is what reaches the process"
hasnt "$out" '^TENSORFOLD_FACES_FP8=all$' "the registry does not overwrite it"

echo "3. an arm with no flags of its own leaves .env exactly as written"
run --exl3-405-turboderp
has "$out" '^TENSORFOLD_FACES_FP8=1$' ".env's value survives"
has "$out" '^TENSORFOLD_EXL3_MIDM=1$' "and so do the EXL3 switches"
has "$out" '/models/exl3-405' "the arm's model replaces .env's"
has "$out" "state-exl3-405-turboderp:/state" "the arm's state directory replaces .env's"

echo "4. the aliases still resolve"
run --exl3
has "$out" '/models/exl3-405' "--exl3 means the 4.05bpw pack"
run --nvfp4-radixart
has "$out" '--nvfp4-radixart is a typo' "the pre-correction spelling is named as such"
has "$out" 'RadixArk/Qwen3.8-Flash-Next-NVFP4' "and selects the RadixArk pack"

echo "5. an arm the registry does not have is refused, not defaulted"
run --nvfp4-typo
refuses "$out" "is not one of" "an unknown arm exits non-zero and says why"
run --nvfp4
refuses "$out" "names no pack" "--nvfp4 alone is refused"

echo "6. a missing registry is a refusal with a way out, not an arm served without its flags"
mv "$tmp/tree/bench/arms.json" "$tmp/tree/bench/arms.json.away"
run --nvfp4-swift
refuses "$out" 'is missing: an arm asked for by flag' "the missing registry is named"
has "$out" 'or serve from .env with no arm flag' "and the way out is stated"
run
has "$out" '/models/from-env' "without a flag, .env still decides"
mv "$tmp/tree/bench/arms.json.away" "$tmp/tree/bench/arms.json"

echo
if (( fails )); then echo "selftest-arms: $fails check(s) failed"; exit 1; fi
echo "selftest-arms: all checks passed"
