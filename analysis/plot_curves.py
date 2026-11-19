"""The one plot: true success per round for the control loop and the experiment loop, plus judged success.

    third_party/openpi/.venv/bin/python analysis/plot_curves.py runs/middle_bowl --out runs/middle_bowl/curves.png

Reads runs/<name>/<judge>/round_*/metrics.json. Error bars are the 95% Wilson intervals from the exam.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir: pathlib.Path) -> list[dict]:
    rows = []
    for m in sorted(run_dir.glob("round_*/metrics.json"), key=lambda p: int(p.parent.name.split("_")[1])):
        d = json.loads(m.read_text())
        if d.get("status") == "ok":
            rows.append(d)
    return rows


def series(rows: list[dict], key: str):
    xs, ys, lo, hi = [], [], [], []
    for r in rows:
        v = r.get(key)
        if v and v.get("rate") is not None:
            xs.append(r["round"]); ys.append(100 * v["rate"])
            lo.append(100 * (v["rate"] - v["ci95"][0])); hi.append(100 * (v["ci95"][1] - v["rate"]))
    return xs, ys, [lo, hi]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_root", type=pathlib.Path, help="runs/<name>, containing sim/ and robometer/")
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    task = None
    sim = load(args.run_root / "sim")
    if sim:
        task = sim[0].get("task")
        x, y, err = series(sim, "true_success")
        ax.errorbar(x, y, yerr=err, marker="o", color="#2a78d6", capsize=3, label="true success, simulator judge (control)")
        x, y, err = series(sim, "judged_success")
        if x:
            ax.errorbar(x, y, yerr=err, marker="s", linestyle=":", color="#2a78d6", alpha=0.6, capsize=3,
                        label="judged success on the control loop (audit only)")
    exp = load(args.run_root / "robometer")
    if exp:
        task = task or exp[0].get("task")
        x, y, err = series(exp, "true_success")
        ax.errorbar(x, y, yerr=err, marker="o", color="#eb6834", capsize=3, label="true success, Robometer judge (experiment)")
        x, y, err = series(exp, "judged_success")
        ax.errorbar(x, y, yerr=err, marker="^", linestyle="--", color="#eb6834", capsize=3, label="judged success, Robometer judge")
    ax.set_xlabel("round")
    ax.set_ylabel("success on the exam (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted({r["round"] for r in sim + exp}))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(args.title or (task or args.run_root.name))
    out = args.out or (args.run_root / "curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
