#!/bin/bash
#SBATCH -J exp-probe
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/probe-%j.out
cd $HOME/scratch/ethan-exp
P=pools/main/scores
S="$P/robometer_robometer-4b-fft-libero_true_agent_f8.jsonl $P/robometer_Robometer-4B_true_agent_f8.jsonl $P/robometer_Robometer-4B_mirror_agent_f8.jsonl $P/roboreward_RoboReward-4B_true_agent_f16.jsonl $P/yesno_Qwen3-VL-4B-Instruct_true_both_f1.jsonl"
for e in siglip dinov2; do echo "######## $e"; robometer/.venv/bin/python -u code/probe.py --pool pools/main --features pools/main/features_$e.npz --scores $S; done
