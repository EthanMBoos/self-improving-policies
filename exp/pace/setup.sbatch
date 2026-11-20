#!/usr/bin/env bash
# One-time install: openpi server env (py3.11, JAX) and LIBERO client env (py3.8).
#SBATCH -J exp-setup
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --exclude=atl1-1-03-011-28-0  # GPU not usable from CUDA or EGL (Nov 2026)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/setup-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
cd "$OPENPI"
GIT_LFS_SKIP_SMUDGE=1 uv sync
[ -d examples/libero/.venv ] || uv venv --python 3.8 examples/libero/.venv
uv pip sync --python "$CLIENT_PY" examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python "$CLIENT_PY" -e packages/openpi-client -e third_party/libero
L="$OPENPI/third_party/libero/libero/libero"
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOC
assets: $L/assets
bddl_files: $L/bddl_files
benchmark_root: $L
datasets: $L/../datasets
init_states: $L/init_files
EOC
nvidia-smi --query-gpu=name,driver_version --format=csv
"$SERVER_PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
PYTHONPATH="$OPENPI/third_party/libero" "$CLIENT_PY" -c "
from libero.libero import benchmark
for s in ['libero_spatial','libero_object','libero_goal','libero_10','libero_90']:
    print(s, benchmark.get_benchmark_dict()[s]().n_tasks)"
echo SETUP_OK
