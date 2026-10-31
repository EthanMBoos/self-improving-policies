#!/usr/bin/env bash
# Append host RAM / GPU memory+utilization every N seconds to a log until killed (PID written to <log>.pid).
# Usage: scripts/rl/sample_resources.sh <log> [interval_s=10] &
LOG="$1"; INTERVAL="${2:-10}"
echo $$ > "$LOG.pid"
while true; do
  echo "$(date +%s) $(free -m | awk '/Mem:/{print "ram_used_mb="$3" ram_avail_mb="$7}') $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print "gpu_mem_mb="$1" gpu_util="$2}')" >> "$LOG"
  sleep "$INTERVAL"
done
