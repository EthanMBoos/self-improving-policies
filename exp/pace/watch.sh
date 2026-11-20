#!/usr/bin/env bash
# Run locally (not on PACE). Polls sacct every 60 s and exits as soon as any of the given jobs reaches a final state,
# printing every job's state, so whoever started it (e.g. Claude in the background) is woken right away.
#
#   exp/pace/watch.sh 7192987 7192931
set -uo pipefail
jobs=$(IFS=,; echo "$*")
ssh_pace() { ssh -S ~/.ssh/ice-ctl eboos3@login-ice.pace.gatech.edu "$@"; }
fails=0
while true; do
  if out=$(ssh_pace "sacct -n -X -P -j $jobs -o JobID,State" 2>&1); then
    fails=0
    if echo "$out" | grep -qE "\|(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)"; then break; fi
  else
    fails=$((fails + 1))
    if [ $fails -ge 5 ]; then echo "WATCH_ERROR: cannot reach PACE (ssh socket down?): $out"; exit 2; fi
  fi
  sleep 60
done
ssh_pace "sacct -X -j $jobs -o JobID,JobName%14,State,Elapsed,ExitCode,NodeList%20; pace-quota | grep -i ^Scratch"
