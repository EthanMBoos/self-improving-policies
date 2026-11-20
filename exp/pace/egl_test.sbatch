#!/usr/bin/env bash
#SBATCH -J exp-egltest
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --exclude=atl1-1-03-011-28-0  # GPU not usable from CUDA or EGL (Nov 2026)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/egltest-%j.out
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
cd /tmp
mine=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | sed 's/^0000//' | tr A-Z a-z)
echo "CVD=$CUDA_VISIBLE_DEVICES mine=$mine all=$(ls /proc/driver/nvidia/gpus | tr '\n' ' ')"
for d in default 0 1 2 3 4 5 6 7; do
  if [ $d = default ]; then
    PYTHONPATH=$OPENPI/third_party/libero timeout 90 $CLIENT_PY $CODE/egl_probe.py 2>&1 | grep -E "^EGL" || echo "EGL default timeout/err"
  else
    CUDA_VISIBLE_DEVICES=$d MUJOCO_EGL_DEVICE_ID=$d PYTHONPATH=$OPENPI/third_party/libero timeout 90 $CLIENT_PY $CODE/egl_probe.py 2>&1 | grep -E "^EGL|Error:" | head -1 || echo "EGL $d timeout"
  fi
done
