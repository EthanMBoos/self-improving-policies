#!/usr/bin/env bash
# Run openpi's LIBERO eval client on all four suites in parallel against a policy server on localhost:8000,
# then print per-task / per-suite success rates.
#
# Usage: scripts/run_libero_eval.sh <out_dir> [trials_per_task=10] [suites...]
# Writes <out_dir>/<suite>.log, <out_dir>/videos_<suite>/ and <out_dir>/summary.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI="$ROOT/third_party/openpi"
OUT="$(realpath -m "${1:?usage: $0 <out_dir> [trials_per_task] [suites...]}")"
TRIALS="${2:-10}"
SUITES=("${@:3}")
[ ${#SUITES[@]} -gt 0 ] || SUITES=(libero_spatial libero_object libero_goal libero_10)

export PYTHONPATH="${PYTHONPATH:-}:$OPENPI/third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_CONFIG_PATH="$ROOT/third_party/libero_config"
PY="$OPENPI/examples/libero/.venv/bin/python"

mkdir -p "$OUT"
cd "$OPENPI"
for s in "${SUITES[@]}"; do
  echo "starting $s ($TRIALS trials/task) -> $OUT/$s.log"
  "$PY" examples/libero/main.py --args.task-suite-name "$s" --args.num-trials-per-task "$TRIALS" \
    --args.video-out-path "$OUT/videos_$s" > "$OUT/$s.log" 2>&1 &
done
wait
python3 "$ROOT/scripts/summarize_libero.py" "$OUT" | tee "$OUT/summary.txt"
