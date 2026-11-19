"""True and judged success per round for every RFT loop, all seeds, from the committed episode tables.

    python3 analysis/plot_seeds.py            # needs matplotlib; writes results/rft_curves_all_seeds.png

Each point pools a policy's 100-attempt exam and its 300 practice attempts: 400 layouts it never trained on.
True is LIBERO's check. Judged is the share the loop's grader approved at the config's threshold.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rft.metrics import wilson  # noqa: E402

RESULTS = REPO / "results"
SEEDS = {"seed 1": "rft_middle_bowl", "seed 2": "rft_middle_bowl_s2", "seed 3": "rft_middle_bowl_s3"}
PANELS = [
    ("Simulator judges (control)", "sim", SEEDS),
    ("Robometer judges (experiment)", "robometer", SEEDS),
    ("LIBERO fine-tune of Robometer judges", "robometer", {"seed 1": "rft_middle_bowl_indomain"}),
]
MARKERS = {"seed 1": "o", "seed 2": "s", "seed 3": "^"}
TRUE, JUDGED = "#2a78d6", "#eb6834"
INK, MUTED, GRID, AXIS, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def pooled(folder: str, loop: str) -> list[tuple[float, float, float, float]]:
    """Per checkpoint 0 to 3: true rate, its 95% interval, judged rate."""
    with (RESULTS / folder / "episodes.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["loop"] == loop]
    out = []
    for c in range(4):
        s = [r for r in rows if r["checkpoint"] == str(c)]
        graded = [r for r in s if r["approved"] != ""]
        assert len(s) == len(graded) == 400, (folder, loop, c, len(s), len(graded))  # 100 exam + 300 practice, all graded
        k = sum(r["sim_success"] == "1" for r in s)
        lo, hi = wilson(k, len(s))
        out.append((k / len(s), lo, hi, sum(r["approved"] == "1" for r in graded) / len(graded)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=pathlib.Path, default=RESULTS / "rft_curves_all_seeds.png")
    args = p.parse_args()

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": AXIS})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, facecolor=SURFACE)
    x = list(range(4))
    for ax, (title, loop, seeds) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for seed, folder in seeds.items():
            pts = pooled(folder, loop)
            true = [100 * t for t, _, _, _ in pts]
            err = [[100 * (t - lo) for t, lo, _, _ in pts], [100 * (hi - t) for t, _, hi, _ in pts]]
            ax.errorbar(x, true, yerr=err, color=TRUE, marker=MARKERS[seed], markersize=5, linewidth=2,
                        elinewidth=1, capsize=0)
            ax.plot(x, [100 * j for *_, j in pts], color=JUDGED, marker=MARKERS[seed], markersize=5, linewidth=2,
                    linestyle="--")
            if seed == "seed 2":
                ax.annotate("seed 2", (3, true[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                            color=INK)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks(x, ["start", "1", "2", "3"])
        ax.set_xlim(-0.3, 3.7)
        ax.set_xlabel("after round")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("success over 400 attempts (%)")

    handles = [
        Line2D([], [], color=TRUE, linewidth=2, label="true success (LIBERO's check, 95% interval)"),
        Line2D([], [], color=JUDGED, linewidth=2, linestyle="--", label="judged success (the loop's grader at 0.65)"),
        *(Line2D([], [], color=MUTED, marker=m, linestyle="", label=s) for s, m in MARKERS.items()),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
