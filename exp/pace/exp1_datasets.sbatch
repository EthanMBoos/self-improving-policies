#!/usr/bin/env bash
# Build Experiment 1 LeRobot datasets from names files, in parallel on the node's local disk, then copy to scratch.
#
#   code/pace/submit.sh code/pace/exp1_datasets.sbatch --export=ALL,POOL=pools/exp1_train,SETS="clean fp30 grader matched"
#
# Reads $EXP/exp1/sets/<set>.txt, writes $EXP/lerobot/exp1/<set>.
#SBATCH -J exp1-data
#SBATCH -p coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp1-data-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
scratch_check || exit 4
: "${POOL:?}" "${SETS:?}"
LOCAL="${TMPDIR:-/tmp}/lerobot-$SLURM_JOB_ID"
mkdir -p "$EXP/lerobot/exp1"
cd "$OPENPI"
for s in $SETS; do
  ( HF_LEROBOT_HOME="$LOCAL" "$SERVER_PY" "$CODE/make_dataset.py" --pool "$EXP/$POOL" \
      --names "$EXP/exp1/sets/$s.txt" --repo-id "exp1/$s" 2>&1 | grep -E "^wrote|Error|Traceback"
    rm -rf "$EXP/lerobot/exp1/$s" && cp -r "$LOCAL/exp1/$s" "$EXP/lerobot/exp1/" && echo "copied $s" ) &
done
wait
rm -rf "$LOCAL"
du -sh "$EXP"/lerobot/exp1/*
echo DATASETS_DONE
