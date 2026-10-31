#!/usr/bin/env bash
# Evaluate checkpoints in LIBERO, one at a time: start the policy server, roll out the four suites in parallel with
# scripts/libero_rollouts.py (per-episode videos / actions / success flags), stop the server, summarize.
# Checkpoints whose <out_dir>/<name>/summary.json exists are skipped, so the script can be re-run to resume.
#
# Usage: scripts/eval_checkpoints.sh <config> <out_dir> <trials_per_task> <checkpoint_dir>...
#   e.g. scripts/eval_checkpoints.sh pi05_libero_lora outputs/eval/sft_5k 5 checkpoints/pi05_libero_lora/sft_5k/*/
# Env: SUITES="libero_spatial libero_object libero_goal libero_10" (default), NO_VIDEO=1 to skip videos.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI="$ROOT/third_party/openpi"
CONFIG="${1:?usage: $0 <config> <out_dir> <trials_per_task> <checkpoint_dir>...}"
OUT="$(realpath -m "$2")"
TRIALS="$3"
shift 3
read -r -a SUITES <<< "${SUITES:-libero_spatial libero_object libero_goal libero_10}"

export PYTHONPATH="${PYTHONPATH:-}:$OPENPI/third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_CONFIG_PATH="$ROOT/third_party/libero_config"
CLIENT_PY="$OPENPI/examples/libero/.venv/bin/python"
SERVER_PY="$OPENPI/.venv/bin/python"

SERVER=""
cleanup() {
  if [ -n "$SERVER" ]; then
    kill "$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for CK in "$@"; do
  CK="$(realpath "$CK")"
  [ -d "$CK/params" ] || { echo "skip $CK (no params/)"; continue; }
  NAME="$(basename "$CK")"
  D="$OUT/$NAME"
  if [ -f "$D/summary.json" ]; then echo "skip $NAME (done)"; continue; fi
  mkdir -p "$D"
  echo "== $NAME: serving $CK"
  "$SERVER_PY" "$ROOT/scripts/serve_policy.py" --config "$CONFIG" --dir "$CK" > "$D/server.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 900); do
    grep -q "server listening" "$D/server.log" && break
    kill -0 "$SERVER" 2>/dev/null || { echo "server failed, see $D/server.log"; exit 1; }
    sleep 1
  done
  grep -q "server listening" "$D/server.log" || { echo "server not ready after 15 min"; exit 1; }

  START=$(date +%s)
  PIDS=()
  for s in "${SUITES[@]}"; do
    (cd "$OPENPI" && "$CLIENT_PY" "$ROOT/scripts/libero_rollouts.py" --suite "$s" --trials-per-task "$TRIALS" \
      --out "$D" --tag "$NAME" ${NO_VIDEO:+--no-video} > "$D/client_$s.log" 2>&1) &
    PIDS+=($!)
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
  cleanup; SERVER=""
  [ "$FAIL" = 0 ] || { echo "a client failed, see $D/client_*.log"; exit 1; }
  echo "{\"eval_wall_s\": $(( $(date +%s) - START )), \"trials_per_task\": $TRIALS, \"checkpoint\": \"$CK\"}" > "$D/eval_info.json"
  python3 "$ROOT/scripts/summarize_rollouts.py" "$D" | tee "$D/summary.txt" | tail -1
done
