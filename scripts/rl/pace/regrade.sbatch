#!/usr/bin/env bash
# Re-grade a finished RL run's recorded episodes with Robometer in several frame orientations, then compare each
# against LIBERO's success check. Grading only: no simulator, no training. See docs/regrade-orientation.md.
#
#   sbatch --export=ALL,RUN=$HOME/scratch/rl_runs/oracle scripts/rl/pace/regrade.sbatch                  # pilot
#   sbatch --export=ALL,RUN=$HOME/scratch/rl_runs/robometer_soft,SAMPLE=0,FLIPS="none lr" scripts/rl/pace/regrade.sbatch
#
# RUN    run directory with episodes.jsonl and episodes/iter_*/<id>/agentview.mp4 (required)
# SAMPLE episodes to grade, chosen at random with seed 0 (default 1000; 0 = all)
# FLIPS  orientations to try (default "none lr ud rot180"; lr is Robometer's training view)
#SBATCH -J regrade-orientation
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=regrade-%j.out
set -euo pipefail
source "${SIP_REPO:-$HOME/scratch/self-improving-policies}/scripts/rl/pace/env.sh"
cd "$SIP_REPO"
: "${RUN:?set RUN to the run directory}"
SAMPLE="${SAMPLE:-1000}"
FLIPS="${FLIPS:-none lr ud rot180}"
NAME="$(basename "$RUN")_n${SAMPLE}"
OUT="$SIP_REPO/outputs/rl/offline/$NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PY=(uv run -q --no-project --with omegaconf --with numpy python)
for flip in $FLIPS; do
  echo "==== $NAME flip=$flip $(date)"
  "${PY[@]}" scripts/rl/grade_offline.py "$RUN" "$OUT/$flip" --grader robometer_soft --flip "$flip" \
    --sample "$SAMPLE" --seed 0 --batch-size 16
done

LOGGED=()
# Robometer runs carry the scores it gave during training (no flip): add them as a row and check reproducibility.
if python3 -c "import json,sys; e=json.loads(open(sys.argv[1]).readline()); sys.exit(e.get('grader_score') is None)" "$RUN/episodes.jsonl"; then
  LOGGED=(--logged "$RUN")
fi
echo "==== comparison $(date)"
dirs=(); for flip in $FLIPS; do dirs+=("$OUT/$flip"); done
"${PY[@]}" scripts/rl/compare_orientations.py "${dirs[@]}" "${LOGGED[@]}" --json "$OUT/comparison.json" | tee "$OUT/comparison.txt"
echo "wrote $OUT/comparison.txt"
