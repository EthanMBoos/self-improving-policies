"""Join training metrics with per-checkpoint LIBERO results into a learning curve (CSV + figure).

Usage (openpi env, needs matplotlib):
    third_party/openpi/.venv/bin/python scripts/learning_curve.py \
        --run checkpoints/pi05_libero_lora/sft_5k --eval outputs/eval/sft_5k --out results/sft_5k

Writes <out>/learning_curve.csv (one row per evaluated checkpoint), <out>/learning_curve.png and copies
metrics.jsonl / run_info.json / data_split.json from the run.
"""

import argparse
import csv
import json
import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
COLORS = {"libero_spatial": "#2a78d6", "libero_object": "#eb6834", "libero_goal": "#1baf7a", "libero_10": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def nearest(records, step):
    return min(records, key=lambda r: abs(r["step"] - step)) if records else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=pathlib.Path, required=True)
    p.add_argument("--eval", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--ref", type=float, action="append", default=[], help="reference average success (%%) lines")
    p.add_argument("--ref-label", action="append", default=[])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = [json.loads(l) for l in (args.run / "metrics.jsonl").read_text().splitlines()]
    train = [m for m in metrics if m["kind"] == "train"]
    val = [m for m in metrics if m["kind"] == "val"]

    rows = []
    for d in sorted((x for x in args.eval.iterdir() if (x / "summary.json").exists() and x.name.isdigit()),
                    key=lambda x: int(x.name)):
        s = json.loads((d / "summary.json").read_text())
        step = int(d.name) + 1  # openpi labels a checkpoint with the 0-based index of its last update
        t, v = nearest(train, step), nearest(val, step)
        row = {
            "checkpoint": d.name,
            "updates": step,
            "samples_seen": t.get("samples_seen"),
            "epochs": t.get("epochs"),
            "gpu_hours": t.get("gpu_hours"),
            "train_loss": t.get("loss"),
            "val_flow_loss": v.get("val/flow_loss"),
            "train_subset_flow_loss": v.get("train_subset/flow_loss"),
            "val_action_l1": v.get("val/action_l1"),
            "train_subset_action_l1": v.get("train_subset/action_l1"),
            "eval_episodes": s["pooled"][1],
            "avg_success": s["average_over_suites"],
        }
        for suite in SUITES:
            row[f"{suite}_success"] = s["suites"].get(suite, {}).get("success_rate")
        rows.append(row)

    with open(args.out / "learning_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["checkpoint"])
        w.writeheader()
        w.writerows(rows)
    for name in ("metrics.jsonl", "run_info.json", "data_split.json"):
        if (args.run / name).exists():
            shutil.copy(args.run / name, args.out / name)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK2)
        ax.tick_params(colors=INK2)
        ax.set_xlabel("training updates (batch 32)", color=INK2)

    ax = axes[0]
    xs = [r["updates"] for r in rows]
    for suite in SUITES:
        ys = [100 * r[f"{suite}_success"] for r in rows if r[f"{suite}_success"] is not None]
        if ys:
            ax.plot(xs[: len(ys)], ys, color=COLORS[suite], linewidth=2, marker="o", markersize=5, label=suite)
    if rows:
        avg = [100 * r["avg_success"] for r in rows]
        ax.plot(xs, avg, color=INK, linewidth=2.5, marker="o", markersize=6, label="average")
        ax.annotate(f"{avg[-1]:.1f}%", (xs[-1], avg[-1]), textcoords="offset points", xytext=(6, -3), color=INK)
    for ref, label in zip(args.ref, args.ref_label):
        ax.axhline(ref, color=INK2, linestyle="--", linewidth=1)
        ax.text(0.01, ref + 1, label, transform=ax.get_yaxis_transform(), color=INK2, fontsize=8, va="bottom")
    ax.set_ylim(-3, 103)
    ax.set_ylabel("LIBERO success rate (%)", color=INK2)
    ax.set_title("Success rate per checkpoint", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot([m["step"] for m in train], [m["loss"] for m in train], color="#2a78d6", linewidth=1, alpha=0.5,
            label="train loss (augmented, per 10 steps)")
    ax.plot([m["step"] for m in val], [m["train_subset/flow_loss"] for m in val], color="#2a78d6", linewidth=2,
            marker="o", markersize=4, label="train demos (fixed frames)")
    ax.plot([m["step"] for m in val], [m["val/flow_loss"] for m in val], color="#eb6834", linewidth=2,
            marker="o", markersize=4, label="held-out demos")
    ax.set_yscale("log")
    ax.set_ylabel("flow-matching loss", color=INK2)
    ax.set_title("Loss: train vs held-out demos", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot([m["step"] for m in val], [m["train_subset/action_l1"] for m in val], color="#2a78d6", linewidth=2,
            marker="o", markersize=4, label="train demos")
    ax.plot([m["step"] for m in val], [m["val/action_l1"] for m in val], color="#eb6834", linewidth=2,
            marker="o", markersize=4, label="held-out demos")
    ax.set_ylabel("sampled action-chunk L1 (normalized)", color=INK2)
    ax.set_title("Action error vs demo actions", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out / "learning_curve.png", dpi=150, facecolor="#fcfcfb")
    print(f"wrote {args.out}/learning_curve.csv, learning_curve.png ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
