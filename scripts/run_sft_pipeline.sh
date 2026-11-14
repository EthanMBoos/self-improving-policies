#!/usr/bin/env bash
# LoRA SFT baseline end to end: train pi0.5 LoRA on the LIBERO demos, evaluate every checkpoint, build the learning
# curve. Every stage resumes / skips finished work, so the script can simply be re-run after an interruption.
#
# Usage: scripts/run_sft_pipeline.sh [exp_name=sft_5k] [steps=5000] [trials_per_ckpt=5] [final_trials=10]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="${1:-sft_5k}"
STEPS="${2:-5000}"
TRIALS="${3:-5}"
FINAL_TRIALS="${4:-10}"
CONFIG=pi05_libero_lora
RUN="$ROOT/checkpoints/$CONFIG/$EXP"
PY="$ROOT/third_party/openpi/.venv/bin/python"
LAST=$((STEPS - 1))

echo "== train ($(date))"
if [ -d "$RUN/$LAST/params" ]; then
  echo "training already finished"
else
  RESUME=""
  [ -f "$RUN/metrics.jsonl" ] && RESUME="--resume"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "$PY" "$ROOT/scripts/train.py" --config "$CONFIG" --exp-name "$EXP" \
    --num-train-steps "$STEPS" $RESUME
fi

echo "== evaluate checkpoints, $TRIALS trials/task ($(date))"
CKPTS=$(find "$RUN" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -t/ -k1 -V)
# shellcheck disable=SC2086
"$ROOT/scripts/eval_checkpoints.sh" "$CONFIG" "$ROOT/outputs/eval/$EXP" "$TRIALS" $CKPTS

echo "== final checkpoint, $FINAL_TRIALS trials/task ($(date))"
"$ROOT/scripts/eval_checkpoints.sh" "$CONFIG" "$ROOT/outputs/eval/${EXP}_final${FINAL_TRIALS}" "$FINAL_TRIALS" "$RUN/$LAST"

echo "== learning curve ($(date))"
"$PY" "$ROOT/scripts/learning_curve.py" --run "$RUN" --eval "$ROOT/outputs/eval/$EXP" --out "$ROOT/results/$EXP" \
  --ref 97.2 --ref-label "pi05_libero (full FT, 10 trials/task)"
echo "== done ($(date))"
