"""Summarize openpi examples/libero/main.py logs: per-task and per-suite success rates.

Usage: python3 scripts/summarize_libero.py <eval_dir>   (expects <suite>.log files; stdlib only)
"""

import pathlib
import re
import sys

PUBLISHED = {"libero_spatial": 98.8, "libero_object": 98.2, "libero_goal": 98.0, "libero_10": 92.4}

d = pathlib.Path(sys.argv[1])
rows = []
for suite in PUBLISHED:
    log = d / f"{suite}.log"
    if not log.exists():
        continue
    task, per_task = None, {}
    for line in log.read_text().splitlines():
        if m := re.search(r"Task: (.+)$", line):
            task = m.group(1)
        elif m := re.search(r"Success: (True|False)", line):
            per_task.setdefault(task, []).append(m.group(1) == "True")
    n = sum(len(v) for v in per_task.values())
    s = sum(sum(v) for v in per_task.values())
    rows.append((suite, s, n))
    print(f"\n== {suite}: {s}/{n} = {100 * s / max(n, 1):.1f}%  (published, 50 trials/task: {PUBLISHED[suite]})")
    for t, v in per_task.items():
        print(f"   {sum(v):>3}/{len(v):<3} {t}")
S, N = sum(r[1] for r in rows), sum(r[2] for r in rows)
print(f"\nAverage over suites: {sum(100 * s / n for _, s, n in rows) / max(len(rows), 1):.1f}%   pooled: {S}/{N}")
