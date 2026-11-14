#!/usr/bin/env bash
# Launch a grader-in-the-loop RL run (RLinf env). Training uses our fresh LIBERO layouts
# (third_party/libero_train_init_states), never the official evaluation layouts.
#
# Usage: scripts/rl/run_rl.sh <run_name> [hydra overrides...]
#   e.g. scripts/rl/run_rl.sh oracle sip.grader.name=oracle sip.min_episodes=4000
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RLINF="$ROOT/third_party/RLinf"
RUN="${1:?usage: $0 <run_name> [overrides...]}"; shift

# LIBERO config for training: same assets as the official one, init states replaced by our training layouts.
TRAIN_CFG="$ROOT/third_party/rlinf_libero_config_train"
mkdir -p "$TRAIN_CFG"
sed "s|^init_states:.*|init_states: $ROOT/third_party/libero_train_init_states|" \
  "$ROOT/third_party/rlinf_libero_config/config.yaml" > "$TRAIN_CFG/config.yaml"

export SIP_REPO="$ROOT"
export EMBODIED_PATH="$RLINF/examples/embodiment"
export PYTHONPATH="$ROOT:$RLINF:${PYTHONPATH:-}"
export RLINF_EXT_MODULE=sip_rl.rlinf_ext
export LIBERO_CONFIG_PATH="$TRAIN_CFG"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
# Ray's memory monitor counts the ZFS ARC cache as used memory and kills workers; the kernel reclaims ARC.
export RAY_memory_monitor_refresh_ms=0

OUT="$ROOT/outputs/rl/runs/$RUN"
mkdir -p "$OUT"
if [ -f "$OUT/run_end.json" ]; then
  echo "run $RUN already finished ($OUT/run_end.json)"; exit 0
fi
# Resume from the latest full checkpoint if the run was interrupted.
LATEST=$(ls -d "$OUT"/rlinf/"$RUN"/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)
RESUME=()
if [ -n "$LATEST" ]; then
  echo "resuming $RUN from $LATEST"
  RESUME=("runner.resume_dir=$LATEST")
fi
"$ROOT/scripts/rl/sample_resources.sh" "$OUT/resources.log" 10 &
trap 'kill $(cat "$OUT/resources.log.pid") 2>/dev/null || true' EXIT
cd "$RLINF"
"$RLINF/.venv/bin/python" "$ROOT/scripts/rl/train_rl.py" sip.run_name="$RUN" "${RESUME[@]}" "$@" 2>&1 | tee -a "$OUT/train.log"
exit "${PIPESTATUS[0]}"
