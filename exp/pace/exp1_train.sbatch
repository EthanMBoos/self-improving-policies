#!/usr/bin/env bash
# One Experiment 1 fine-tune: pi05_libero_lora on $EXP/lerobot/exp1/$RUN, same settings for every row.
#
#   code/pace/submit.sh code/pace/exp1_train.sbatch --export=ALL,RUN=clean[,STEPS=2000,SEED=42,EXP_NAME=clean]
#
# The dataset is built from $EXP/exp1/sets/$RUN.txt on the node's local disk (10-15 min) unless it already exists
# in $EXP/lerobot, so datasets never take scratch space. Checkpoints go to
# $EXP/checkpoints/pi05_libero_lora/$EXP_NAME/<step>; only params (~6 GB) are kept, train_state is deleted at the end.
# Only the final checkpoint is saved (save interval = STEPS): six jobs saving mid-run copies at once overflow scratch.
# SEED defaults to openpi's 42, which the first round used.
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
STEPS="${STEPS:-2000}"; SEED="${SEED:-42}"; EXP_NAME="${EXP_NAME:-$RUN}"; POOL="${POOL:-pools/exp1_train}"
export PYTHONPATH="$CODE"
# the datasets library copies every dataset it loads into its cache (~9 GB each); keep that copy on local disk
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf-datasets-$SLURM_JOB_ID"
trap 'rm -rf "$HF_DATASETS_CACHE"' EXIT
cd "$OPENPI"
if [ -f "$EXP/lerobot/exp1/$RUN/meta/info.json" ]; then
  export HF_LEROBOT_HOME="$EXP/lerobot"
else
  export HF_LEROBOT_HOME="${TMPDIR:-/tmp}/lerobot-$SLURM_JOB_ID"
  trap 'rm -rf "$HF_LEROBOT_HOME" "$HF_DATASETS_CACHE"' EXIT
  t0=$(date +%s)
  "$SERVER_PY" "$CODE/make_dataset.py" --pool "$EXP/$POOL" --names "$EXP/exp1/sets/$RUN.txt" --repo-id "exp1/$RUN" \
    2>&1 | grep -E "^wrote|Error|Traceback"
  echo "dataset built in $(( $(date +%s) - t0 ))s"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "run=$RUN name=$EXP_NAME seed=$SEED steps=$STEPS dataset=$(python3 -c "import json; d=json.load(open('$HF_LEROBOT_HOME/exp1/$RUN/meta/info.json')); print(d['total_episodes'], 'episodes,', d['total_frames'], 'frames')")"
t0=$(date +%s)
# python -u and no grep buffering, so step lines appear as they happen
PYTHONUNBUFFERED=1 "$SERVER_PY" "$CODE/train.py" pi05_libero_lora --exp-name "$EXP_NAME" --data.repo-id "exp1/$RUN" \
  --seed "$SEED" --num-train-steps "$STEPS" --log-interval 50 --save-interval "$STEPS" --num-workers 8 --overwrite 2>&1 \
  | grep --line-buffered -E "^Step|Saving checkpoint|Error|Traceback" \
  | while IFS= read -r l; do echo "$(date +%T) $l"; done
echo "train wall time $(( $(date +%s) - t0 ))s"
rm -rf "$EXP/checkpoints/pi05_libero_lora/$EXP_NAME"/*/train_state
ls "$EXP/checkpoints/pi05_libero_lora/$EXP_NAME/"
echo TRAIN_DONE
