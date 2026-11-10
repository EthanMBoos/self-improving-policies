"""Compare Robometer's agreement with LIBERO's success check across frame orientations.

Each argument is an output directory of scripts/rl/grade_offline.py (one per --flip). All are scored on the episodes
they share, so a --sample pilot compares every orientation on the same episodes. Needs numpy only:

    uv run --no-project --with numpy python scripts/rl/compare_orientations.py outputs/rl/offline/pilot_*  \
        [--logged outputs/rl/runs/robometer_soft] [--threshold 0.5] [--json out.json]

--logged adds the scores Robometer gave during that RL run (no flip) as an extra row, and checks that the offline
"none" variant reproduces them.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sip_rl.grader_metrics import agreement  # noqa: E402


def load(path: pathlib.Path) -> dict:
    return {e["id"]: e for e in map(json.loads, (path / "episodes.jsonl").read_text().splitlines())}


def best_balanced_accuracy(score, oracle):
    best = (0.0, None)
    for t in np.linspace(0.05, 0.95, 19):
        b = agreement(oracle, (score > t).astype(int))["balanced_accuracy"]
        if b is not None and b > best[0]:
            best = (b, float(t))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="+", type=pathlib.Path)
    p.add_argument("--logged", type=pathlib.Path, help="RL run directory whose in-loop grader scores to add")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--json", type=pathlib.Path)
    a = p.parse_args()

    rows = {}
    for d in a.variants:
        src = json.loads((d / "source.json").read_text()) if (d / "source.json").exists() else {}
        rows[f"{d.name} (flip {src.get('flip', '?')})"] = load(d)
    if a.logged:
        rows["logged during RL (no flip)"] = load(a.logged)

    ids = sorted(set.intersection(*(set(r) for r in rows.values())))
    if not ids:
        sys.exit("the directories share no episode ids")
    oracle = np.array([int(next(iter(rows.values()))[i]["oracle_success_at_end"]) for i in ids])

    results = {}
    print(f"{len(ids)} shared episodes, true success {oracle.mean():.1%}, threshold {a.threshold}\n")
    head = f"{'variant':<40} {'says ok':>8} {'prec':>6} {'recall':>7} {'FPR':>6} {'AUROC':>6} {'best bal acc':>13}"
    print(head)
    print("-" * len(head))
    for name, eps in rows.items():
        score = np.array([float(eps[i]["grader_score"]) for i in ids])
        m = agreement(oracle, (score > a.threshold).astype(int), score)
        bal, t = best_balanced_accuracy(score, oracle)
        m["best_balanced_accuracy"], m["best_threshold"] = bal, t
        results[name] = {k: v for k, v in m.items() if k != "per_task"}
        print(f"{name:<40} {m['grader_success_rate']:>8.1%} {m['precision'] or 0:>6.2f} {m['recall'] or 0:>7.2f} "
              f"{m['false_positive_rate'] or 0:>6.2f} {m['auroc'] or 0:>6.3f} {bal:>8.3f} @{t:.2f}")

    none = [n for n in rows if n.endswith("(flip none)")]
    if a.logged and none:
        x = np.array([float(rows[none[0]][i]["grader_score"]) for i in ids])
        y = np.array([float(rows["logged during RL (no flip)"][i]["grader_score"]) for i in ids])
        diff = float(np.abs(x - y).max())
        print(f"\nreproducibility: offline 'none' vs logged scores, max abs diff {diff:.4f}, "
              f"correlation {np.corrcoef(x, y)[0, 1]:.4f}")
        results["reproducibility"] = {"max_abs_diff": diff, "correlation": float(np.corrcoef(x, y)[0, 1])}

    if a.json:
        a.json.write_text(json.dumps({"episodes": len(ids), "true_success": float(oracle.mean()),
                                      "threshold": a.threshold, "results": results}, indent=2))


if __name__ == "__main__":
    main()
