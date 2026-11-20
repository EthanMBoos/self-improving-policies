#!/usr/bin/env bash
# Probe env creation + rendering for a list of tasks, one process each, with a timeout.
#   sbatch --export=ALL,SUITE=libero_10,TASKS="0 1 2" code/pace/task_probe.sbatch
#SBATCH -J exp-taskprobe
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --exclude=atl1-1-03-011-28-0  # GPU not usable from CUDA or EGL (Nov 2026)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/taskprobe-%j.out
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
cd /tmp
for t in $TASKS; do
  PYTHONPATH=$OPENPI/third_party/libero timeout 120 $CLIENT_PY $CODE/egl_probe.py $SUITE $t > /tmp/probe_$$.log 2>&1
  rc=$?
  grep -h PROBE /tmp/probe_$$.log || { echo "PROBE $SUITE/$t rc=$rc"; grep -v WARNING /tmp/probe_$$.log | tail -6; }
done
