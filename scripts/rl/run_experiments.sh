#!/usr/bin/env bash
# Full experiment, one run after another on one GPU: for each (run name, grader preset) train with PPO, evaluate
# its snapshots, compute the offline goal report and the grader analysis. Every stage resumes or skips finished
# work, so re-running this script after an interruption continues where it stopped.
#
# Usage: scripts/rl/run_experiments.sh [run:grader ...]
#   default: oracle:oracle robometer_soft:robometer_soft robometer_bin:robometer
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || RUNS=(oracle:oracle robometer_soft:robometer_soft robometer_bin:robometer)

for spec in "${RUNS[@]}"; do
  RUN="${spec%%:*}"; GRADER="${spec##*:}"
  echo "==== $RUN (grader $GRADER) $(date)"
  "$ROOT/scripts/rl/run_rl.sh" "$RUN" sip.grader_name="$GRADER"
  echo "==== $RUN: evaluation $(date)"
  python3 "$ROOT/scripts/rl/eval_run.py" "$RUN"
  echo "==== $RUN: goal report + grader analysis $(date)"
  LIBERO_CONFIG_PATH="$ROOT/third_party/rlinf_libero_config_train" MUJOCO_GL=egl PYTHONPATH="$ROOT" \
    "$ROOT/third_party/RLinf/.venv/bin/python" "$ROOT/scripts/rl/goal_report.py" "$ROOT/outputs/rl/runs/$RUN"
  "$ROOT/third_party/openpi/.venv/bin/python" "$ROOT/scripts/rl/analyze_grader.py" "$ROOT/outputs/rl/runs/$RUN"
done
echo "==== all done $(date)"
