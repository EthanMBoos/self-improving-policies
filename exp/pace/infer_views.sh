#!/usr/bin/env bash
# Make inference copies of the pinned Robometer snapshots with unsloth turned off (config.yaml: use_unsloth: false).
# The weights are symlinked; only config.yaml changes. With unsloth on, the current unsloth patch drops
# hidden_states and Robometer's forward fails.
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
for m in robometer--Robometer-4B jesbu1--robometer-4b-fft-libero; do
  src=$(ls -d "$HF_HOME"/hub/models--$m/snapshots/*/ | head -1)
  dst="$EXP/ckpt/${m#*--}"
  mkdir -p "$dst"
  for f in "$src"*; do b=$(basename "$f"); [ "$b" = config.yaml ] || ln -sfn "$(readlink -f "$f")" "$dst/$b"; done
  sed 's/^\(\s*\)use_unsloth: true/\1use_unsloth: false/' "$src/config.yaml" > "$dst/config.yaml"
  echo "$dst from $src: $(grep use_unsloth "$dst/config.yaml")"
done
