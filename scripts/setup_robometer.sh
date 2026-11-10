#!/usr/bin/env bash
# Set up the Robometer grader: its repository and environment under third_party/, and its weights under checkpoints/.
# Follows the recipe in Stratos's README (branch stratos, 3db6273), steps 2 to 4 of "Grader-in-the-loop RL".
#
#   scripts/setup_robometer.sh            # robometer/Robometer-4B (general; the primary grader, D1-matched split)
#   scripts/setup_robometer.sh --in-domain  # also jesbu1/robometer-4b-fft-libero (the second grader, D4)
#
# Requires: git, uv, network; HF_HOME on the pool (env.sh) so the 8.9 GB download does not land in home.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ROBOMETER_COMMIT=352d160

if [ ! -d third_party/robometer/.git ]; then
  git clone https://github.com/robometer/robometer.git third_party/robometer
fi
git -C third_party/robometer checkout -q "$ROBOMETER_COMMIT"
echo "== robometer env"
# torchao is removed because the version it pulls in fails to import (Stratos, 2026-10-31).
(cd third_party/robometer && uv sync --extra robometer)
(cd third_party/robometer && uv pip uninstall --python .venv/bin/python torchao) || echo "torchao was not installed"

download() {  # <hf repo> <local dir>
  echo "== $1 -> $2"
  HF_HUB_DISABLE_XET=1 uv run --no-project --with huggingface_hub python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('$1', local_dir='$2')"
}
infer_view() {  # <local dir>: a copy of the checkpoint with unsloth turned off for inference
  local src="$1" dst="$1-infer"
  mkdir -p "$dst"
  for f in "$src"/*; do b=$(basename "$f"); [ "$b" = config.yaml ] || ln -sfn "../$(basename "$src")/$b" "$dst/$b"; done
  sed 's/^\(\s*\)use_unsloth: true/\1use_unsloth: false/' "$src/config.yaml" > "$dst/config.yaml"
}

download robometer/Robometer-4B checkpoints/Robometer-4B
infer_view checkpoints/Robometer-4B
if [ "${1:-}" = "--in-domain" ]; then
  download jesbu1/robometer-4b-fft-libero checkpoints/Robometer-4B-fft-libero
  infer_view checkpoints/Robometer-4B-fft-libero
fi

echo "== check"
third_party/robometer/.venv/bin/python -c "import robometer, torch; print('robometer ok, cuda', torch.cuda.is_available())"
echo "Setup done. Grader model dir: checkpoints/Robometer-4B-infer"
