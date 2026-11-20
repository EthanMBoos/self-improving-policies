#!/usr/bin/env bash
#   sbatch --export=ALL,POOL=pools/main code/pace/features.sbatch
#SBATCH -J exp-features
#SBATCH -p coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --exclude=atl1-1-03-011-28-0  # GPU not usable from CUDA or EGL (Nov 2026)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/features-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
scratch_check || exit 4
for e in siglip dinov2; do
  "$EXP/robometer/.venv/bin/python" "$CODE/features.py" --pool "$EXP/$POOL" --encoder $e --out "$EXP/$POOL/features_$e.npz"
done
