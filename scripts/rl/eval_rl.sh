#!/usr/bin/env bash
# Evaluate the start checkpoint or an RL snapshot on LIBERO's OFFICIAL evaluation layouts (RLinf evaluator,
# deterministic sampling, success = task completed at any point within 520 steps).
#
# Usage: scripts/rl/eval_rl.sh <out_dir> <episodes_per_task> [snapshot.pt] [extra hydra overrides...]
#   no snapshot  -> evaluates the start checkpoint (RLinf-Pi05-LIBERO-SFT)
#   snapshot.pt  -> start checkpoint + expert LoRA + trained heads from the snapshot
# Writes <out_dir>/summary.json (overall and per-task success) and eval.log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RLINF="$ROOT/third_party/RLinf"
OUT="$(realpath -m "${1:?usage: $0 <out_dir> <episodes_per_task> [snapshot.pt] [overrides...]}")"
PER_TASK="${2:?episodes per task}"
SNAP="${3:-}"
shift $(( $# >= 3 ? 3 : 2 ))
ENVS="${EVAL_ENVS:-16}"            # parallel envs (each ~0.75 GB GPU, ~1.7 GB RAM)
CHUNK="${EVAL_CHUNK:-10}"          # actions executed per policy call; 10 = the RL training setting
TOTAL=$(( PER_TASK * 10 ))         # LIBERO-10 has 10 tasks
EPOCHS=$(( (TOTAL + ENVS - 1) / ENVS ))

EXTRA=()
if [ -n "$SNAP" ]; then
  EXTRA+=("+rollout.model.expert_lora_rank=${LORA_RANK:-32}" "+rollout.model.trainable_ckpt=$(realpath "$SNAP")")
fi

export EMBODIED_PATH="$RLINF/examples/embodiment" PYTHONPATH="$ROOT:$RLINF:${PYTHONPATH:-}"
export RLINF_EXT_MODULE=sip_rl.rlinf_ext
export LIBERO_CONFIG_PATH="$ROOT/third_party/rlinf_libero_config"   # official init states
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 RAY_memory_monitor_refresh_ms=0
mkdir -p "$OUT"
cd "$RLINF"
START=$(date +%s)
"$RLINF/.venv/bin/python" evaluations/eval_embodied_agent.py --config-path "$RLINF/evaluations/libero" \
  --config-name libero_10_openpi_pi05_eval \
  rollout.model.model_path="$ROOT/checkpoints/RLinf-Pi05-LIBERO-SFT" rollout.model.num_action_chunks="$CHUNK" \
  env.eval.total_num_envs="$ENVS" env.eval.rollout_epoch="$EPOCHS" env.eval.video_cfg.save_video=False \
  runner.logger.log_path="$OUT" "${EXTRA[@]}" "$@" > "$OUT/eval.log" 2>&1
WALL=$(( $(date +%s) - START ))
python3 - "$OUT" "$WALL" "$SNAP" "$CHUNK" <<'EOF'
import collections, json, re, sys
out, wall, snap, chunk = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
per = collections.defaultdict(list)
for line in open(f"{out}/eval.log"):
    m = re.search(r"task_id=(\d+), trial_id=(\d+), success=(True|False)", line)
    if m:
        per[int(m.group(1))].append(m.group(3) == "True")
n = sum(len(v) for v in per.values()); s = sum(sum(v) for v in per.values())
summary = {"snapshot": snap or None, "action_chunk": chunk, "episodes": n, "successes": s,
           "success_rate": s / n if n else None, "wall_s": wall,
           "per_task": {str(t): {"n": len(v), "success_rate": sum(v) / len(v)} for t, v in sorted(per.items())}}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
print(f"{s}/{n} = {100 * s / max(n, 1):.1f}%  ({wall} s)")
EOF
