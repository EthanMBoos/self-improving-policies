# Source this on PACE ICE (login or compute node) before setup or grading. Keeps the big things in ~/scratch,
# since the home quota is small: uv's package cache, the Hugging Face cache, and the repo itself.
export SCRATCH="${SCRATCH:-$HOME/scratch}"
export UV_CACHE_DIR="$SCRATCH/.uv-cache"
export HF_HOME="$SCRATCH/.hf-cache"
export PATH="$HOME/.local/bin:$PATH"
export SIP_REPO="${SIP_REPO:-$SCRATCH/self-improving-policies}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"
command -v uv >/dev/null || echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"
