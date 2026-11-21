#!/usr/bin/env bash
# One Experiment 1 fine-tune: pi05_libero_lora on $EXP/lerobot/exp1/$RUN, same settings for every row.
#
#   code/pace/submit.sh code/pace/exp1_train.sbatch --export=ALL,RUN=clean[,STEPS=2000]
#
# Checkpoints go to $EXP/checkpoints/pi05_libero_lora/$RUN/<step> (openpi keeps the latest, ~9 GB).
# Batch 32 needs ~108 GB of GPU memory, so H200 only.
#SBATCH -J exp1-train
#SBATCH -p coe-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=cuda13
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp1-train-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
scratch_check || exit 4
: "${RUN:?}"
STEPS="${STEPS:-2000}"
export HF_LEROBOT_HOME="$EXP/lerobot" PYTHONPATH="$CODE"
[ -f "$HF_LEROBOT_HOME/exp1/$RUN/meta/info.json" ] || { echo "no dataset exp1/$RUN"; exit 1; }
cd "$OPENPI"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "run=$RUN steps=$STEPS dataset=$(python3 -c "import json; d=json.load(open('$HF_LEROBOT_HOME/exp1/$RUN/meta/info.json')); print(d['total_episodes'], 'episodes,', d['total_frames'], 'frames')")"
t0=$(date +%s)
# python -u and no grep buffering, so step lines appear as they happen
PYTHONUNBUFFERED=1 "$SERVER_PY" "$CODE/train.py" pi05_libero_lora --exp-name "$RUN" --data.repo-id "exp1/$RUN" \
  --num-train-steps "$STEPS" --log-interval 50 --save-interval 1000 --num-workers 8 --overwrite 2>&1 \
  | grep --line-buffered -E "^Step|Saving checkpoint|Error|Traceback" \
  | while IFS= read -r l; do echo "$(date +%T) $l"; done
echo "train wall time $(( $(date +%s) - t0 ))s"
ls "$EXP/checkpoints/pi05_libero_lora/$RUN/"
echo TRAIN_DONE
