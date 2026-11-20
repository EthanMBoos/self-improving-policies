#!/usr/bin/env bash
# Serve a pi0.5 checkpoint and run several LIBERO rollout clients against it in parallel.
#
#   code/pace/submit.sh code/pace/rollouts.sbatch --export=ALL,OUT=pools/probe_l90,SUITE=libero_90,TASKS="0 1 2",INITS="0 1",WORKERS=6 
#
# OUT      output dir, relative to $EXP (required)
# SUITE    LIBERO suite (default libero_90)
# TASKS    task ids (default: all); split across workers
# INITS    init state indices (default "0")
# SEED     env seed (default 7)
# CONFIG   openpi config (default pi05_libero);  CKPT  checkpoint dir (default gs://openpi-assets/checkpoints/$CONFIG)
# TAG      appended to episode names
# WORKERS  parallel sim clients (default 6)
#SBATCH -J exp-rollouts
#SBATCH -p coe-gpu  # without a partition PACE spreads the job over 3 partitions and drops --exclude
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda13
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rollouts-%j.out
set -euo pipefail
source "$HOME/scratch/ethan-exp/code/pace/env.sh"
scratch_check || exit 4
: "${OUT:?set OUT}"
SUITE="${SUITE:-libero_90}"; INITS="${INITS:-0}"; SEED="${SEED:-7}"; WORKERS="${WORKERS:-6}"; TAG="${TAG:-}"
CONFIG="${CONFIG:-pi05_libero}"; CKPT="${CKPT:-gs://openpi-assets/checkpoints/$CONFIG}"
PORT=$((20000 + RANDOM % 20000))

# robosuite uses CUDA_VISIBLE_DEVICES as an index into the EGL device list, which does not follow SLURM's GPU
# numbering, so it can pick a GPU this job does not own (software fallback ~0.4 steps/s, or EGL failure).
# Try the default and then each EGL device index; keep the first that renders fast.
probe_once() {
  (cd /tmp && PYTHONPATH="$OPENPI/third_party/libero" timeout 60 "$CLIENT_PY" "$CODE/egl_probe.py" 2>/dev/null | grep PROBE || true)
}
echo "node $SLURMD_NODENAME, earlier checks:"; node_history
good=""
for dev in default 0 1 2 3 4 5 6 7 8; do
  if [ "$dev" = default ]; then unset MUJOCO_EGL_DEVICE_ID; else export MUJOCO_EGL_DEVICE_ID=$dev; fi
  probe=$(probe_once)
  rate=$(echo "$probe" | grep -oE "[0-9.]+ steps/s" | cut -d" " -f1 || true)
  echo "render probe on $SLURMD_NODENAME, EGL device $dev (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}): ${probe:-timeout}"
  if [ -n "$rate" ] && [ "$(echo "$rate >= 10" | bc)" = 1 ]; then good=$dev; break; fi
  node_log render "bad" "EGL device $dev: ${probe:-timeout}"
done
if [ -z "$good" ]; then
  node_log render "FAIL" "no EGL device rendered fast"
  echo "BAD_RENDER_NODE $SLURMD_NODENAME"
  exit 3
fi
node_log render "ok" "EGL device $good, $rate steps/s"
echo "using EGL device $good"
mkdir -p "$EXP/$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "config=$CONFIG ckpt=$CKPT suite=$SUITE seed=$SEED port=$PORT"

cd "$OPENPI"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5  # leave GPU memory for the simulators' EGL rendering
"$SERVER_PY" "$CODE/serve.py" --port "$PORT" policy:checkpoint --policy.config="$CONFIG" --policy.dir="$CKPT" \
  > "$EXP/$OUT/server-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 600); do
  python3 -c "import socket; socket.create_connection(('127.0.0.1', $PORT), 1)" 2>/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -50 "$EXP/$OUT/server-$SLURM_JOB_ID.log"; exit 1; }
  sleep 2
done
echo "server up after ~$((i * 2))s"
# Compile the model with one dummy request before any simulator touches the GPU (compiling while 8 simulators
# render stalled for >10 min). Give up after 10 min.
t0=$(date +%s)
if ! timeout 600 "$SERVER_PY" - "$PORT" <<'PY'
import functools, sys, numpy as np
import websockets.sync.client as wsc
wsc.connect = functools.partial(wsc.connect, ping_interval=None)  # the server can't answer pings while compiling
from openpi_client import websocket_client_policy as w
c = w.WebsocketClientPolicy("127.0.0.1", int(sys.argv[1]))
img = np.zeros((224, 224, 3), np.uint8)
for _ in range(2):
    c.infer({"observation/image": img, "observation/wrist_image": img, "observation/state": np.zeros(8, np.float32),
             "prompt": "warm up"})
PY
then echo "SERVER_WARMUP_FAILED"; node_log server bad "warm-up inference failed or took over 600 s"; exit 5; fi
echo "server warm-up (compile) took $(( $(date +%s) - t0 ))s"

if [ -z "${TASKS:-}" ]; then
  TASKS=$(PYTHONPATH="$OPENPI/third_party/libero" "$CLIENT_PY" -c \
    "from libero.libero import benchmark; print(*range(benchmark.get_benchmark_dict()['$SUITE']().n_tasks))" | tail -1)
fi
read -ra ALL <<< "$TASKS"
pids=()
for w in $(seq 0 $((WORKERS - 1))); do
  mine=()
  for j in "${!ALL[@]}"; do [ $((j % WORKERS)) -eq "$w" ] && mine+=("${ALL[$j]}"); done
  [ ${#mine[@]} -eq 0 ] && continue
  PYTHONPATH="$OPENPI/third_party/libero" "$CLIENT_PY" "$CODE/rollouts.py" --out "$EXP/$OUT" --suite "$SUITE" \
    --tasks "${mine[@]}" --inits $INITS --seed "$SEED" --port "$PORT" ${TAG:+--tag "$TAG"} \
    > "$EXP/$OUT/worker$w-$SLURM_JOB_ID.log" 2>&1 &
  pids+=($!)
  sleep 3
done
# Every 10 minutes: check scratch; at 95% stop the workers cleanly (finished episodes are kept, a rerun resumes).
( while sleep 600; do
    scratch_check 2>&1 | sed "s/^/[$(date +%T)] /" || { echo "stopping workers: scratch full"; kill "${pids[@]}" 2>/dev/null; exit; }
  done ) &
watch=$!
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
kill $watch 2>/dev/null || true
python3 - "$EXP/$OUT/episodes.jsonl" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rows: by[(r["task_id"], r["task"])].append(r["success"])
for (t, name), s in sorted(by.items()): print("%3d %4.0f%% (%d)  %s" % (t, 100 * sum(s) / len(s), len(s), name))
print("total %d episodes, %.1f%% success" % (len(rows), 100 * sum(r["success"] for r in rows) / len(rows)))
PY
exit $fail
