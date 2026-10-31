"""Evaluate an RL run's snapshots on LIBERO-10's official layouts (after training; resumable).

Usage: python3 scripts/rl/eval_run.py <run_name> [--min-episodes 4096] [--every 1024]
Schedule:
    start checkpoint (shared by all runs)               500 episodes   outputs/rl/evals/start
    snapshots at multiples of --every episodes           200 episodes
    cutoff snapshots (min-episodes reached, >=12 GPU-h,  500 episodes
                      end of run)
Writes outputs/rl/evals/<run>/<snapshot>/summary.json and outputs/rl/evals/<run>/curve.json.
"""

import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


def evaluate(out: pathlib.Path, per_task: int, snapshot: str | None) -> dict:
    if not (out / "summary.json").exists():
        cmd = [str(ROOT / "scripts/rl/eval_rl.sh"), str(out), str(per_task)] + ([snapshot] if snapshot else [])
        subprocess.run(cmd, check=True)
    return json.loads((out / "summary.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--min-episodes", type=int, default=4096)
    p.add_argument("--every", type=int, default=1024)
    p.add_argument("--start-per-task", type=int, default=50)
    p.add_argument("--snapshot-per-task", type=int, default=20)
    p.add_argument("--cutoff-per-task", type=int, default=50)
    p.add_argument("--evals-dir", default="outputs/rl/evals")
    a = p.parse_args()
    run_dir = ROOT / "outputs/rl/runs" / a.run
    evals = ROOT / a.evals_dir
    start = evaluate(evals / "start", a.start_per_task, None)
    points = [{"snapshot": None, "episodes": 0, "gpu_hours": 0.0, "eval_episodes": start["episodes"],
               "success_rate": start["success_rate"], "per_task": start["per_task"]}]
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r.get("snapshot"):
            continue
        reasons = r.get("snapshot_reasons", [])
        cutoff = any(x.startswith("gpu_h>=") for x in reasons) or "end" in reasons or r["episodes"] == a.min_episodes
        if cutoff:
            per_task = a.cutoff_per_task
        elif r["episodes"] % a.every == 0:
            per_task = a.snapshot_per_task
        else:
            continue
        snap = pathlib.Path(r["snapshot"])
        s = evaluate(evals / a.run / f"{snap.stem}_{per_task * 10}", per_task, str(snap))
        points.append({"snapshot": str(snap), "episodes": r["episodes"], "gpu_hours": r["gpu_hours"],
                       "reasons": reasons, "eval_episodes": s["episodes"], "success_rate": s["success_rate"],
                       "per_task": s["per_task"]})
        print(f"{a.run}: {r['episodes']} episodes / {r['gpu_hours']:.1f} GPU-h -> {s['success_rate']:.3f} ({s['episodes']} eval episodes)")
    (evals / a.run).mkdir(parents=True, exist_ok=True)
    (evals / a.run / "curve.json").write_text(json.dumps(points, indent=2))


if __name__ == "__main__":
    main()
