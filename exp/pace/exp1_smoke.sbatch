#!/usr/bin/env bash
# Experiment 1 smoke test: build a small dataset, check a batch, train a few LoRA steps and time them.
#
#   code/pace/submit.sh code/pace/exp1_smoke.sbatch --export=ALL,STEPS=60[,RUN=smoke,WORKERS=2]
#
# Checks, in order (stops at the first failure):
#   1 dataset   the pool's true successes -> LeRobot dataset exp1/smoke_clean
#   2 batch     one batch through openpi's pipeline: shapes, normalized action stats, image orientation (png)
#   3 train     STEPS LoRA steps from pi05_libero (default 60): loss, seconds/step, GPU memory, checkpoint written
#SBATCH -J exp1-smoke
#SBATCH -p coe-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=cuda13
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp1-smoke-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
scratch_check || exit 4
export HF_LEROBOT_HOME="$EXP/lerobot" PYTHONPATH="$CODE"
STEPS="${STEPS:-60}"; RUN="${RUN:-smoke}"; WORKERS="${WORKERS:-2}"
mkdir -p "$EXP/exp1"
cd "$OPENPI"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== 1 dataset"
python3 - "$EXP/pools/main" > "$EXP/exp1/smoke_clean.txt" <<'PY'
import json, pathlib, sys
for p in sorted((pathlib.Path(sys.argv[1]) / "episodes").iterdir()):
    m = json.loads((p / "meta.json").read_text())
    if m["success"]:
        print(p.name)
PY
# LeRobot writes hundreds of small image files per episode; on Lustre scratch that is ~1 min/episode. Build on the
# node's local disk, then copy the finished dataset to scratch.
LOCAL="${TMPDIR:-/tmp}/lerobot-$SLURM_JOB_ID"
t0=$(date +%s)
if [ -f "$HF_LEROBOT_HOME/exp1/smoke_clean/meta/info.json" ] && [ -z "${REBUILD:-}" ]; then
  echo "dataset exists, reusing (set REBUILD=1 to rebuild)"
else
HF_LEROBOT_HOME="$LOCAL" "$SERVER_PY" "$CODE/make_dataset.py" --pool "$EXP/pools/main" \
  --names "$EXP/exp1/smoke_clean.txt" --repo-id exp1/smoke_clean 2>&1 | grep -vE "Map:|parquet from Arrow"
rm -rf "$HF_LEROBOT_HOME/exp1/smoke_clean" && mkdir -p "$HF_LEROBOT_HOME/exp1"
cp -r "$LOCAL/exp1/smoke_clean" "$HF_LEROBOT_HOME/exp1/" && rm -rf "$LOCAL"
echo "dataset built and copied in $(( $(date +%s) - t0 ))s; $(du -sh "$HF_LEROBOT_HOME/exp1/smoke_clean" | cut -f1)"
fi

echo "== 2 batch"
timeout 900 "$SERVER_PY" "$CODE/check_batch.py" exp1/smoke_clean "$EXP/exp1/smoke_batch.png" 2>&1 | grep -vE "^WARNING|rocm|tpu"

echo "== 3 train $STEPS steps"
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 20 > "$EXP/exp1/smoke_gpu_mem.txt" &
MON=$!
t0=$(date +%s)
timeout 2700 "$SERVER_PY" "$CODE/train.py" pi05_libero_lora --exp-name "$RUN" --data.repo-id exp1/smoke_clean \
  --num-train-steps "$STEPS" --log-interval 10 --save-interval "$STEPS" --num-workers "$WORKERS" --overwrite 2>&1 \
  | grep -vE "^WARNING|rocm|tpu" | grep -E "^Step|Saving checkpoint|Error|Traceback" \
  | while IFS= read -r l; do echo "$(date +%T) $l"; done
kill $MON 2>/dev/null || true
echo "train wall time $(( $(date +%s) - t0 ))s; peak GPU memory $(sort -n "$EXP/exp1/smoke_gpu_mem.txt" | tail -1)"
ls "$EXP/checkpoints/pi05_libero_lora/$RUN/"
echo SMOKE_DONE
