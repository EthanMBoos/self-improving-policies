#!/usr/bin/env bash
# Set up openpi (policy server) and the LIBERO simulator (eval client) under third_party/.
#
#   third_party/openpi/.venv                    Python 3.11 + JAX: serves the policy on the GPU
#   third_party/openpi/examples/libero/.venv    Python 3.8: LIBERO / robosuite simulation client
#   third_party/libero_config/config.yaml       project-local LIBERO paths (used via LIBERO_CONFIG_PATH,
#                                               so an existing ~/.libero from another project is left alone)
#
# Requires: git, uv, an NVIDIA GPU + driver (tested: RTX 5090, driver 580, CUDA 13.0).
set -euo pipefail

OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479  # openpi main as of 2026-10-19
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI="$ROOT/third_party/openpi"

if [ ! -d "$OPENPI/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
fi
cd "$OPENPI"
git checkout -q "$OPENPI_COMMIT"
git submodule update --init --recursive

echo "== openpi server env"
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "== LIBERO client env"
[ -d examples/libero/.venv ] || uv venv --python 3.8 examples/libero/.venv
CLIENT_PY=examples/libero/.venv/bin/python
uv pip sync --python "$CLIENT_PY" \
  examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python "$CLIENT_PY" -e packages/openpi-client -e third_party/libero

echo "== LIBERO config"
L="$OPENPI/third_party/libero/libero/libero"
mkdir -p "$ROOT/third_party/libero_config"
cat > "$ROOT/third_party/libero_config/config.yaml" <<EOF
assets: $L/assets
bddl_files: $L/bddl_files
benchmark_root: $L
datasets: $L/../datasets
init_states: $L/init_files
EOF

echo "== checks"
.venv/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
PYTHONPATH="$OPENPI/third_party/libero" LIBERO_CONFIG_PATH="$ROOT/third_party/libero_config" \
  "$CLIENT_PY" -c "from libero.libero import benchmark; import openpi_client; print('libero client ok')"
echo "Setup done."
