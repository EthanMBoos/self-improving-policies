#!/usr/bin/env bash
# One-time install of the grader env (official Robometer repo, which brings torch + transformers with Qwen3-VL)
# and download of the pinned grader checkpoints.
#SBATCH -J exp-setup-graders
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --exclude=atl1-1-03-011-28-0  # GPU not usable from CUDA or EGL (Nov 2026)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup-graders-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
cd "$EXP/robometer"
uv sync
uv pip install --python .venv/bin/python scikit-learn imageio imageio-ffmpeg
for m in robometer/Robometer-4B jesbu1/robometer-4b-fft-libero teetone/RoboReward-4B Qwen/Qwen3-VL-4B-Instruct; do
  .venv/bin/hf download "$m" > /dev/null && echo "downloaded $m @ $(ls $HF_HOME/hub/models--${m//\//--}/snapshots)"
done
.venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name())"
echo SETUP_OK
