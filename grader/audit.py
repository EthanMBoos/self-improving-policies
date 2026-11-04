"""Confusion matrix of the grader against the simulator over a rollout directory (build-order step 7).

Pure Python; runs on any machine that has the sidecar. Reads <rollout_dir>/episodes_extra.jsonl after grading
(grader/robometer.py), sweeps the thresholds, applies the precision floor (D5), and lists the false positives
to watch. Writes <rollout_dir>/audit.json and audit.md.

    python3 grader/audit.py runs/x/sim/round_1/rollouts --config configs/rft/middle_bowl.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rft import metrics, sidecar  # noqa: E402
from rft.config import LoopConfig  # noqa: E402


def audit(records: list[dict], thresholds: list[float], default_threshold: float, floor: float, top: int = 5) -> dict:
    graded = [r for r in records if not r.get("error") and r.get("grader_score") is not None]
    if not graded:
        raise ValueError("no graded episodes; run the grader first")
    sim = [int(bool(r["sim_success"])) for r in graded]
    scores = [float(r["grader_score"]) for r in graded]
    sweep = metrics.threshold_sweep(sim, scores, sorted(set(thresholds) | {default_threshold}))
    at_default = sweep[str(default_threshold)]
    precision = at_default["precision"]
    verdict = {
        "threshold": default_threshold,
        "precision": precision,
        "floor": floor,
        "passes_floor": (precision is not None and precision >= floor),
        "note": ("no approvals at this threshold" if precision is None else
                 "experiment loop may run" if precision >= floor else
                 "below the floor: switch to the in-domain checkpoint or fine-tune the grader first"),
    }
    failures = sorted((r for r in graded if not r["sim_success"]), key=lambda r: -float(r["grader_score"]))
    successes = sorted((r for r in graded if r["sim_success"]), key=lambda r: float(r["grader_score"]))
    pick = lambda rs: [{"id": r["id"], "seed": r["seed"], "grader_score": r["grader_score"], "n_steps": r["n_steps"],  # noqa: E731
                        "video": str(pathlib.Path(r["dir"]) / "agentview.mp4")} for r in rs[:top]]
    return {
        "n_graded": len(graded),
        "n_ungraded": len(records) - len(graded),
        "sim_success": metrics.rate_with_ci(sum(sim), len(sim)),
        "grader_name": graded[0].get("grader_name"),
        "sweep": sweep,
        "verdict": verdict,
        "false_positives_top": pick(failures),
        "false_negatives_top": pick(successes),
        "score_histogram": _histogram(scores, sim),
    }


def _histogram(scores: list[float], sim: list[int], bins: int = 10) -> list[dict]:
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if lo <= s < hi or (hi == 1.0 and s == 1.0)]
        rows.append({"bin": [lo, hi], "n": len(idx), "n_sim_success": sum(sim[i] for i in idx)})
    return rows


def to_markdown(a: dict) -> str:
    lines = [f"# Grader audit: {a['grader_name']} on {a['n_graded']} episodes", ""]
    ss = a["sim_success"]
    lines.append(f"Simulator success {ss['k']}/{ss['n']} = {100 * ss['rate']:.1f}% [{100 * ss['ci95'][0]:.0f}, {100 * ss['ci95'][1]:.0f}]")
    lines += ["", "| threshold | tp | fp | fn | tn | precision | recall | fpr | grader rate | gap |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, m in a["sweep"].items():
        f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        lines.append(f"| {t} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['false_positive_rate'])} | {f(m['grader_success_rate'])} | {f(m.get('success_rate_gap'))} |")
    any_m = next(iter(a["sweep"].values()))
    if any_m.get("auroc") is not None:
        lines += ["", f"AUROC {any_m['auroc']:.3f}, Brier {any_m['brier']:.3f}, ECE {any_m['ece']:.3f}"]
    v = a["verdict"]
    prec = "-" if v["precision"] is None else f"{v['precision']:.2f}"
    outcome = "PASS" if v["passes_floor"] else "FAIL"
    lines += ["", f"**D5 at threshold {v['threshold']}: precision {prec} versus floor {v['floor']}: {outcome}.** {v['note']}", ""]
    lines += ["## False positives to watch (simulator failed, highest grader score)", ""]
    for r in a["false_positives_top"]:
        lines.append(f"- {r['id']} seed {r['seed']}: score {r['grader_score']:.3f}, {r['n_steps']} steps, `{r['video']}`")
    lines += ["", "## False negatives (simulator succeeded, lowest grader score)", ""]
    for r in a["false_negatives_top"]:
        lines.append(f"- {r['id']} seed {r['seed']}: score {r['grader_score']:.3f}, {r['n_steps']} steps, `{r['video']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollout_dir", type=pathlib.Path)
    p.add_argument("--config", type=pathlib.Path, default=None, help="loop config; supplies thresholds")
    p.add_argument("--thresholds", type=float, nargs="*", default=None)
    p.add_argument("--threshold", type=float, default=None, help="the pass mark the loop uses")
    p.add_argument("--floor", type=float, default=0.5, help="precision floor (D5)")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()
    thresholds, default = [0.5, 0.65, 0.85], 0.65
    if args.config:
        g = LoopConfig.load(args.config).grader
        thresholds, default = list(g.audit_thresholds), g.threshold
    if args.thresholds:
        thresholds = args.thresholds
    if args.threshold is not None:
        default = args.threshold
    records = sidecar.read_records(args.rollout_dir / sidecar.FILENAME)
    a = audit(records, thresholds, default, args.floor, args.top)
    (args.rollout_dir / "audit.json").write_text(json.dumps(a, indent=2))
    md = to_markdown(a)
    (args.rollout_dir / "audit.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
