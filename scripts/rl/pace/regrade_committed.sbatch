#!/usr/bin/env bash
# Quick orientation check on the labelled rollout videos already in git (about 96: the pi0.5 baseline evals and the
# LIBERO-90 candidate clips). Grades each in four orientations and compares against LIBERO's success check, for all
# videos and for the hard subset without the zero-shot base model's failures. See docs/regrade-orientation.md.
#
#   sbatch scripts/rl/pace/regrade_committed.sbatch
#SBATCH -J regrade-committed
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=regrade-committed-%j.out
set -euo pipefail
source "${SIP_REPO:-$HOME/scratch/self-improving-policies}/scripts/rl/pace/env.sh"
cd "$SIP_REPO"
FLIPS="${FLIPS:-none lr ud rot180}"
RUN="$SIP_REPO/outputs/rl/committed"
OUT="$SIP_REPO/outputs/rl/offline/committed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 scripts/rl/videos_to_episodes.py "$RUN" results/libero_eval/*/videos_* results/libero90_sweep_50/videos_candidates

PY=(uv run -q --no-project --with omegaconf --with numpy python)
for flip in $FLIPS; do
  echo "==== flip=$flip $(date)"
  "${PY[@]}" scripts/rl/grade_offline.py "$RUN" "$OUT/$flip" --grader robometer_soft --flip "$flip" --batch-size 16
done

dirs=(); for flip in $FLIPS; do dirs+=("$OUT/$flip"); done
{
  echo "==== all committed videos"
  "${PY[@]}" scripts/rl/compare_orientations.py "${dirs[@]}" --json "$OUT/comparison_all.json"
  echo
  echo "==== hard subset: without the zero-shot base model, whose arm just drifts away"
  "${PY[@]}" scripts/rl/compare_orientations.py "${dirs[@]}" --source '^(pi05_libero|libero90)' --json "$OUT/comparison_hard.json"
} | tee "$OUT/comparison.txt"
echo "wrote $OUT/comparison.txt"
