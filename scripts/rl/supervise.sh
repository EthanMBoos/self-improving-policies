#!/usr/bin/env bash
# Re-launch run_experiments.sh after a crash (e.g. a rare CUDA OOM from allocator fragmentation). Every stage resumes:
# training continues from the run's latest checkpoint (saved every 512 episodes), finished runs/evals are skipped.
# Optionally first waits for an already running pipeline (its PID) to exit.
#
# Usage: scripts/rl/supervise.sh [max_launches=3] [pid_to_wait_for] [run:grader ...]
# Logs: outputs/rl/experiments.log (pipeline, appended), outputs/rl/supervisor.log
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAX="${1:-3}"; WAIT_PID="${2:-}"; shift $(( $# >= 2 ? 2 : $# ))
LOG="$ROOT/outputs/rl/experiments.log"; SUP="$ROOT/outputs/rl/supervisor.log"
say() { echo "[supervisor $(date '+%F %T')] $*" | tee -a "$SUP"; }
START=$(cat "$LOG" 2>/dev/null | wc -l)   # only this supervisor's launches count (the log is appended across stages)
finished() { tail -n +$((START + 1)) "$LOG" 2>/dev/null | tail -n 1 | grep -q "^==== all done"; }

ran=false
if [ -n "$WAIT_PID" ]; then
  say "waiting for running pipeline pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  ran=true
fi
launches=0
until finished; do
  $ran && say "pipeline stopped before finishing; last error: $(grep -E 'Error|error:' "$LOG" | tail -1 | cut -c1-200)"
  if [ "$launches" -ge "$MAX" ]; then say "giving up after $launches launches"; exit 1; fi
  launches=$((launches + 1)); ran=true
  say "launching pipeline (launch $launches of $MAX)"
  "$ROOT/scripts/rl/run_experiments.sh" "$@" >> "$LOG" 2>&1
done
say "pipeline finished"
