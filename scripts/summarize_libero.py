"""Summarize openpi examples/libero/main.py logs: per-task and per-suite success rates.

Usage: python3 scripts/summarize_libero.py <eval_dir> [--window LO HI]
Reads every <suite>.log in <eval_dir> (stdlib only). Tasks are sorted by success rate and
those inside the window (default 0.10 to 0.40) are marked, which is how held-out tasks are picked.
"""

from __future__ import annotations

import argparse
import pathlib
import re

PUBLISHED = {"libero_spatial": 98.8, "libero_object": 98.2, "libero_goal": 98.0, "libero_10": 92.4}
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def parse_log(log: pathlib.Path) -> dict[str, list[bool]]:
    task, per_task = None, {}
    for line in log.read_text().splitlines():
        if m := re.search(r"Task: (.+)$", line):
            task = m.group(1)
        elif m := re.search(r"Success: (True|False)", line):
            per_task.setdefault(task, []).append(m.group(1) == "True")
    return per_task


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_dir", type=pathlib.Path)
    p.add_argument("--window", nargs=2, type=float, default=(0.10, 0.40), metavar=("LO", "HI"))
    args = p.parse_args()
    lo, hi = args.window

    logs = {log.stem: log for log in args.eval_dir.glob("*.log")}
    suites = [s for s in SUITE_ORDER if s in logs] + sorted(s for s in logs if s not in SUITE_ORDER)
    rows, in_window = [], []
    for suite in suites:
        per_task = parse_log(logs[suite])
        n = sum(len(v) for v in per_task.values())
        s = sum(sum(v) for v in per_task.values())
        rows.append((suite, s, n))
        pub = f"  (published, 50 trials/task: {PUBLISHED[suite]})" if suite in PUBLISHED else ""
        print(f"\n== {suite}: {s}/{n} = {100 * s / max(n, 1):.1f}%{pub}")
        for t, v in sorted(per_task.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), kv[0])):
            rate = sum(v) / len(v)
            mark = "  <-- window" if lo <= rate <= hi else ""
            print(f"   {sum(v):>3}/{len(v):<3} {100 * rate:5.1f}%  {t}{mark}")
            if mark:
                in_window.append((suite, rate, t))
    S, N = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"\nAverage over suites: {sum(100 * s / n for _, s, n in rows if n) / max(len(rows), 1):.1f}%   pooled: {S}/{N}")
    print(f"\nTasks in the {100 * lo:.0f} to {100 * hi:.0f}% window: {len(in_window)}")
    for suite, rate, t in sorted(in_window, key=lambda x: x[1]):
        print(f"   {100 * rate:5.1f}%  {suite}  {t}")


if __name__ == "__main__":
    main()
