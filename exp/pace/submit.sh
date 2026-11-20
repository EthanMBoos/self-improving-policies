#!/usr/bin/env bash
# Submit a job, skipping every node that node_health.tsv marks FAIL (jobs append to it when a node can't do CUDA or render).
#
#   code/pace/submit.sh code/pace/rollouts.sbatch --export=ALL,OUT=pools/main,...
#
# To give a node another chance, delete its FAIL lines from $EXP/node_health.tsv.
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
scratch_check || exit 1
bad=$(awk -F'\t' '$6 == "FAIL" {print $3}' "$EXP/node_health.tsv" 2>/dev/null | sort -u | paste -sd, -)
echo "skipping nodes: ${bad:-none}" >&2
cd "$EXP"
script="$1"; shift
sbatch -p coe-gpu ${bad:+--exclude="$bad"} "$@" "$script"  # options must come before the script
