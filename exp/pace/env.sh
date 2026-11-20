# Paths for the experiments on PACE. Source this from every job.
export EXP="$HOME/scratch/ethan-exp"
export CODE="$EXP/code"
export OPENPI="$EXP/openpi"
export UV_CACHE_DIR="$EXP/.cache/uv"
export HF_HOME="$EXP/.cache/hf"
export OPENPI_DATA_HOME="$EXP/.cache/openpi"
export LIBERO_CONFIG_PATH="$EXP/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CLIENT_PY="$OPENPI/examples/libero/.venv/bin/python"
export SERVER_PY="$OPENPI/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"

# One line per node check, shared by all jobs, so bad nodes are on record: $EXP/node_health.tsv
# columns: time  job  node  gpu  check  result  detail
node_log() {
  local gpu; gpu=$(nvidia-smi --query-gpu=name,pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr -s ' ' || echo "?")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "${SLURM_JOB_ID:-}" "${SLURMD_NODENAME:-$(hostname -s)}" \
    "$gpu" "$1" "$2" "${3:-}" >> "$EXP/node_health.tsv"
}
# Print what node_health.tsv says about this node before running (PACE ignores --exclude, so we can only warn and fail fast).
node_history() {
  [ -f "$EXP/node_health.tsv" ] && grep -P "\t${SLURMD_NODENAME}\t" "$EXP/node_health.tsv" | tail -5 | sed 's/^/  history: /' || true
}

# Scratch quota (300 GB, 1M files). Warn at 85%, refuse to start at 95% of either, so jobs never die on a full disk.
scratch_check() {
  local line gb_pct file_pct
  line=$(pace-quota 2>/dev/null | grep -i "^Scratch") || { echo "scratch_check: pace-quota unavailable" >&2; return 0; }
  read -r gb_pct file_pct <<< "$(echo "$line" | awk '{print $4, $7}' | tr -d %)"
  echo "scratch: $(echo "$line" | awk '{print $2 " of " $3 " GB (" $4 "), " $5 " files (" $7 ")"}')" >&2
  if [ "${gb_pct%.*}" -ge 95 ] || [ "${file_pct%.*}" -ge 95 ]; then
    echo "SCRATCH_FULL: refusing to run. Free space in ~/scratch first." >&2; return 1
  elif [ "${gb_pct%.*}" -ge 85 ] || [ "${file_pct%.*}" -ge 85 ]; then
    echo "WARNING: SCRATCH NEARLY FULL (${gb_pct}% GB, ${file_pct}% files)" >&2
  fi
}
