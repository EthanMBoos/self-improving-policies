#!/usr/bin/env bash
# Grade a pool with several grader settings, one after another, on one GPU.
#
#   code/pace/submit.sh code/pace/grade.sbatch --export=ALL,POOL=pools/main,SETTINGS="robometer:robometer/Robometer-4B:true:agent"
#
# SETTINGS  space-separated grader:model:view:camera[:frames]  (see grade.py)
# NAMES     optional file of episode names, relative to $EXP
#SBATCH -J exp-grade
#SBATCH -p coe-gpu  # without a partition PACE spreads the job over 3 partitions and drops --exclude
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/grade-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
scratch_check || exit 4
: "${POOL:?}" "${SETTINGS:?}"
PY="$EXP/robometer/.venv/bin/python"
nvidia-smi --query-gpu=name --format=csv,noheader
# Some nodes hand out a GPU CUDA cannot open ("busy or unavailable"). Check before loading any model and fail fast.
echo "node $SLURMD_NODENAME, earlier checks:"; node_history
if err=$("$PY" -c "import torch; torch.zeros(1).cuda(); print(torch.cuda.get_device_name())" 2>&1 | tail -1) \
   && [[ "$err" != *rror* ]]; then
  node_log cuda ok "$err"
else
  node_log cuda FAIL "$err"
  echo "BAD_CUDA_NODE $SLURMD_NODENAME: $err"
  exit 3
fi
for s in $SETTINGS; do
  IFS=: read -r grader model view camera frames <<< "$s"
  frames="${frames:-8}"
  tag="${grader}_$(basename "$model")_${view}_${camera}_f${frames}"
  echo "==== $tag $(date)"
  "$PY" "$CODE/grade.py" --pool "$EXP/$POOL" --grader "$grader" --model "$model" --view "$view" --camera "$camera" \
    --frames "$frames" ${NAMES:+--names "$EXP/$NAMES"} --out "$EXP/$POOL/scores/$tag.jsonl"
done
echo GRADE_DONE
