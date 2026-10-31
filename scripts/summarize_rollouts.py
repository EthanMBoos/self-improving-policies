"""Summarize scripts/libero_rollouts.py output: success rates per suite / task, episode lengths (stdlib only).

Usage: python3 scripts/summarize_rollouts.py <eval_dir>    (expects <eval_dir>/<suite>/episodes.jsonl)
Prints a table and writes <eval_dir>/summary.json.
"""

import collections
import json
import pathlib
import statistics
import sys

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

d = pathlib.Path(sys.argv[1])
summary = {"suites": {}}
for suite in SUITES + sorted(p.name for p in d.iterdir() if p.is_dir() and p.name not in SUITES):
    log = d / suite / "episodes.jsonl"
    if not log.exists():
        continue
    eps = [json.loads(line) for line in log.read_text().splitlines()]
    tasks = collections.defaultdict(list)
    for e in eps:
        tasks[(e["task_id"], e["task"])].append(e)
    n, s = len(eps), sum(e["success"] for e in eps)
    succ_steps = [e["steps"] for e in eps if e["success"]]
    summary["suites"][suite] = {
        "episodes": n,
        "successes": s,
        "success_rate": s / n if n else None,
        "mean_steps_success": statistics.mean(succ_steps) if succ_steps else None,
        "errors": sum(1 for e in eps if e.get("error")),
        "per_task": {f"{tid:02d} {task}": [sum(e["success"] for e in v), len(v)] for (tid, task), v in sorted(tasks.items())},
    }
    print(f"\n== {suite}: {s}/{n} = {100 * s / n:.1f}%   mean steps (successes): "
          f"{summary['suites'][suite]['mean_steps_success'] or float('nan'):.0f}")
    for k, (ks, kn) in summary["suites"][suite]["per_task"].items():
        print(f"   {ks:>3}/{kn:<3} {k}")

rates = [v["success_rate"] for k, v in summary["suites"].items() if k in SUITES and v["success_rate"] is not None]
summary["average_over_suites"] = sum(rates) / len(rates) if rates else None
summary["pooled"] = [sum(v["successes"] for v in summary["suites"].values()), sum(v["episodes"] for v in summary["suites"].values())]
if rates:
    print(f"\nAverage over suites: {100 * summary['average_over_suites']:.1f}%   pooled: {summary['pooled'][0]}/{summary['pooled'][1]}")
(d / "summary.json").write_text(json.dumps(summary, indent=2))
