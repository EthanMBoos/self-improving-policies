"""True and judged success per round for every seed of the RFT loops, one panel per seed.

Reads results/rft_middle_bowl{,_s2,_s3}/episodes.csv. Each point is a policy's 400 attempts on layouts it never
trained on (its exam plus its practice set): LIBERO's check ("true") and the share the general Robometer-4B approved
at 0.65 ("judged"). Shaded bands are 95% Wilson intervals on the true rate.

    uv run --no-project --with matplotlib python analysis/plot_all_seeds.py [--out results/rft_all_seeds.png]
"""

import argparse
import collections
import csv
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEEDS = [("Seed 1", "rft_middle_bowl"), ("Seed 2", "rft_middle_bowl_s2"), ("Seed 3", "rft_middle_bowl_s3")]
LOOPS = [("sim", "Simulator as judge", "#2a78d6"), ("robometer", "Robometer as judge", "#eb6834")]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(folder):
    agg = collections.defaultdict(lambda: [0, 0, 0])  # n, true successes, approved
    with open(ROOT / "results" / folder / "episodes.csv") as f:
        for r in csv.DictReader(f):
            a = agg[(r["loop"], int(r["checkpoint"]))]
            a[0] += 1
            a[1] += int(r["sim_success"])
            a[2] += int(r["approved"])
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "rft_all_seeds.png")
    a = p.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
    rounds = [0, 1, 2, 3]
    for ax, (title, folder) in zip(axes, SEEDS):
        agg = load(folder)
        ax.set_facecolor(SURFACE)
        for loop, label, color in LOOPS:
            n = [agg[(loop, k)][0] for k in rounds]
            true = [100 * agg[(loop, k)][1] / agg[(loop, k)][0] for k in rounds]
            judged = [100 * agg[(loop, k)][2] / agg[(loop, k)][0] for k in rounds]
            lo, hi = zip(*(wilson(agg[(loop, k)][1], m) for k, m in zip(rounds, n)))
            ax.fill_between(rounds, [100 * x for x in lo], [100 * x for x in hi], color=color, alpha=0.12, lw=0)
            ax.plot(rounds, judged, color=color, lw=2, ls=(0, (4, 3)), marker="o", ms=5, mfc=SURFACE, mew=1.5)
            ax.plot(rounds, true, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=2)
            ax.annotate(f"{int(true[-1] + 0.5)}%", (3, true[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=10, fontweight="bold")
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_xticks(rounds)
        ax.set_xticklabels(["Start", "1", "2", "3"])
        ax.set_xlabel("Rounds of fine-tuning")
        ax.set_xlim(-0.2, 3.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Success, % of 400 attempts")

    handles = [plt.Line2D([], [], color=color, lw=6, label=label) for _, label, color in LOOPS]
    handles += [plt.Line2D([], [], color=INK2, lw=2, label="True success (LIBERO's check)"),
                plt.Line2D([], [], color=INK2, lw=2, ls=(0, (4, 3)), label="Robometer's reading")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.045, 0.99),
               labelcolor=INK2, fontsize=9.5, handlelength=2.4, columnspacing=2.2)
    fig.text(0.05, 0.015, "Solid: LIBERO's success check. Dashed: share Robometer-4B approves at 0.65. "
             "Bands: 95% intervals on true success.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(a.out, dpi=180, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
