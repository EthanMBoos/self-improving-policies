#!/usr/bin/env bash
# Backfill fixture joint positions into scene.json by replaying saved actions (no rendering), 8 shards in parallel.
#
#   code/pace/submit.sh code/pace/replay_joints.sbatch --export=ALL,POOL=pools/main
#SBATCH -J exp-replay
#SBATCH -p coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/replay-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
: "${POOL:?}"
cd /tmp
for s in 0 1 2 3 4 5 6 7; do
  PYTHONPATH="$OPENPI/third_party/libero" "$CLIENT_PY" "$CODE/replay_joints.py" --pool "$EXP/$POOL" --shard $s --shards 8 \
    2>&1 | grep -E "success=|done|Error|Traceback" &
done
wait
