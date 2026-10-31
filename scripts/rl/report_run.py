"""Print a readable report of an RL run's metrics (outputs/rl/runs/<run>/metrics.jsonl + episodes.jsonl).

Usage: python3 scripts/rl/report_run.py outputs/rl/runs/<run> [--all-keys]
"""

import argparse
import json
import pathlib


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != 0 and abs(v) < 1e-2:
            return f"{v:.2e}"
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("--all-keys", action="store_true")
    a = p.parse_args()
    recs = [json.loads(line) for line in (a.run_dir / "metrics.jsonl").read_text().splitlines()]
    print(f"run: {a.run_dir.name}   grader: {recs[0]['grader']}   iterations: {len(recs)}")

    print("\n== progress & compute (per iteration)")
    cols = ["iteration", "episodes", "env_steps", "gradient_steps", "gpu_hours", "iteration_s", "sync_weights_s",
            "rollout_s", "grading_s", "update_s", "gpu_mem_used_mb", "snapshot_reasons"]
    print(" | ".join(cols))
    for r in recs:
        print(" | ".join(fmt(r.get(c)) for c in cols))

    print("\n== policy performance during training (oracle labels of the training episodes)")
    for r in recs:
        print(f"iter {r['iteration']}: success_at_end={fmt(r['train/oracle_success_rate'])} "
              f"success_once={fmt(r['train/oracle_success_once_rate'])} per_task={r['train/per_task_oracle_success']}")

    print("\n== grader vs oracle (training episodes)")
    keys = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "balanced_accuracy",
            "auroc", "brier", "ece", "grader_success_rate", "oracle_success_rate", "success_rate_gap"]
    print("iter | " + " | ".join(keys))
    for r in recs:
        g = r["grader_vs_oracle"]
        print(f"{r['iteration']} | " + " | ".join(fmt(g.get(k)) for k in keys))
    print("\n== reward the policy received (split by true outcome)")
    for r in recs:
        print(f"iter {r['iteration']}: mode={r.get('reward_mode')} {r.get('reward')} "
              f"steps_to_success={fmt(r.get('train/steps_to_success_once'))}")
    if recs[-1].get("grader_timing"):
        print("grader timing (last iteration):", recs[-1]["grader_timing"])

    print("\n== RL optimization (actor training metrics, last iteration)")
    for k, v in sorted(recs[-1].get("train", {}).items()):
        print(f"  train/{k} = {fmt(v)}")
    for k, v in sorted(recs[-1].get("rollout", {}).items()):
        print(f"  rollout/{k} = {fmt(v)}")
    print("\n== env metrics (last iteration)")
    for k, v in sorted(recs[-1].get("env", {}).items()):
        print(f"  env/{k} = {fmt(v)}")

    eps = [json.loads(line) for line in (a.run_dir / "episodes.jsonl").read_text().splitlines()]
    print(f"\n== per-episode records: {len(eps)}; fields: {sorted(eps[0])}")
    print("example:", json.dumps({k: v for k, v in eps[0].items() if k != "grader_extra"}))
    if a.all_keys:
        print("\nall metric keys:", sorted(recs[-1]))
    end = a.run_dir / "run_end.json"
    if end.exists():
        print("\nrun end:", end.read_text())


if __name__ == "__main__":
    main()
