"""Quantitative + qualitative analysis of a grader against the oracle over an RL run.

Usage (openpi env, has numpy / matplotlib / imageio):
    third_party/openpi/.venv/bin/python scripts/rl/analyze_grader.py outputs/rl/runs/<run> [--window 512] [--gallery 12]

Reads <run>/episodes.jsonl (+ goal_report.jsonl from scripts/rl/goal_report.py if present) and writes <run>/analysis/:
    summary.json      agreement overall / per training window / per task, reward statistics, threshold sweep,
                      calibration, partial-completion and success-then-undo breakdowns
    *.png             figures (see FIGURES below)
    gallery.html      most confident false positives / false negatives per task: frames, Robometer curves, goals
"""

import argparse
import collections
import html
import json
import pathlib
import sys

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sip_rl.grader_metrics import agreement, auroc, expected_calibration_error  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2)


def threshold_sweep(score, oracle):
    rows = []
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        a = agreement(oracle, (score > t).astype(int))
        rows.append({"threshold": float(t), **{k: a[k] for k in ("precision", "recall", "false_positive_rate",
                                                                    "f1", "balanced_accuracy", "grader_success_rate")}})
    return rows


def reliability(score, oracle, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (score >= lo) & ((score < hi) if hi < 1 else (score <= hi))
        if m.any():
            out.append({"bin": [float(lo), float(hi)], "n": int(m.sum()), "mean_score": float(score[m].mean()),
                        "oracle_success_rate": float(oracle[m].mean())})
    return out


def frame_strip(video, n=8):
    frames = iio.imread(video)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return np.concatenate([frames[i] for i in idx], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=pathlib.Path)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--gallery", type=int, default=12, help="examples per error type")
    a = p.parse_args()
    out = a.run / "analysis"
    out.mkdir(exist_ok=True)

    eps = [json.loads(line) for line in (a.run / "episodes.jsonl").read_text().splitlines()]
    goals = {}
    if (a.run / "goal_report.jsonl").exists():
        goals = {g["id"]: g for g in map(json.loads, (a.run / "goal_report.jsonl").read_text().splitlines())}
    oracle = np.array([int(e["oracle_success_at_end"]) for e in eps])
    once = np.array([int(bool(e["oracle_success_once"])) for e in eps])
    label = np.array([int(e["grader_label"]) for e in eps])
    score = np.array([e["grader_score"] if e["grader_score"] is not None else e["grader_label"] for e in eps], float)
    reward = np.array([e.get("reward", e["grader_label"]) for e in eps], float)
    tasks = np.array([e["task_id"] for e in eps])

    summary = {"run": a.run.name, "episodes": len(eps), "overall": agreement(oracle, label, score, tasks.tolist())}

    # Over training: windows of equal episode counts.
    windows = []
    for start in range(0, len(eps), a.window):
        sl = slice(start, start + a.window)
        w = agreement(oracle[sl], label[sl], score[sl])
        rf, rs = reward[sl][oracle[sl] == 0], reward[sl][oracle[sl] == 1]
        windows.append({"episodes": [start, min(start + a.window, len(eps))], **w,
                        "reward_mean_on_failure": float(rf.mean()) if len(rf) else None,
                        "reward_mean_on_success": float(rs.mean()) if len(rs) else None})
    summary["windows"] = windows
    summary["threshold_sweep"] = threshold_sweep(score, oracle)
    summary["reliability"] = reliability(score, oracle)
    summary["ece"] = expected_calibration_error(score, oracle)
    summary["auroc"] = auroc(score, oracle)

    # Success then undo: completed at some point but not at the end.
    undo = (once == 1) & (oracle == 0)
    summary["success_then_undo"] = {"n": int(undo.sum()), "grader_label_rate": float(label[undo].mean()) if undo.any() else None,
                                    "mean_score": float(score[undo].mean()) if undo.any() else None}

    # Partial completion: score / label rate by number of goal predicates satisfied at the end.
    part = collections.defaultdict(list)
    for e, s, l in zip(eps, score, label):
        g = goals.get(e["id"])
        if g:
            part[(g["n_satisfied"], g["n_goals"])].append((s, l, e["task_id"]))
    summary["by_goals_satisfied"] = {
        f"{k[0]}/{k[1]}": {"n": len(v), "mean_score": float(np.mean([x[0] for x in v])),
                           "grader_label_rate": float(np.mean([x[1] for x in v]))}
        for k, v in sorted(part.items())
    }
    if goals:
        summary["restored_state_check"] = {
            "episodes": len(goals), "agree_with_logged_oracle": sum(g["restored_success"] == g["oracle_success_at_end"] for g in goals.values())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- figures ----------------
    figs = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    x = [w["episodes"][1] for w in windows]
    axes[0].plot(x, [w["oracle_success_rate"] for w in windows], color=BLUE, marker="o", lw=2, label="true (oracle)")
    axes[0].plot(x, [w["grader_success_rate"] for w in windows], color=ORANGE, marker="o", lw=2, label="grader says")
    style(axes[0], "Success rate of training episodes", "training episodes", "success rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, [w["false_positive_rate"] for w in windows], color=ORANGE, marker="o", lw=2, label="false-positive rate")
    axes[1].plot(x, [w["false_negative_rate"] for w in windows], color=BLUE, marker="o", lw=2, label="false-negative rate")
    style(axes[1], "Grader error rates over training", "training episodes", "rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(x, [w["reward_mean_on_failure"] for w in windows], color=ORANGE, marker="o", lw=2, label="on true failures")
    axes[2].plot(x, [w["reward_mean_on_success"] for w in windows], color=BLUE, marker="o", lw=2, label="on true successes")
    style(axes[2], "Mean reward the policy received", "training episodes", "reward")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "over_training.png", dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    figs.append("over_training.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    bins = np.linspace(0, 1, 21)
    axes[0].hist(score[oracle == 0], bins=bins, color=ORANGE, alpha=0.7, label=f"true failures (n={int((oracle == 0).sum())})")
    axes[0].hist(score[oracle == 1], bins=bins, color=BLUE, alpha=0.7, label=f"true successes (n={int((oracle == 1).sum())})")
    style(axes[0], "Grader score by true outcome", "grader score (success probability)", "episodes")
    axes[0].legend(frameon=False, fontsize=8)
    rel = summary["reliability"]
    axes[1].plot([0, 1], [0, 1], color=INK2, ls="--", lw=1)
    axes[1].plot([r["mean_score"] for r in rel], [r["oracle_success_rate"] for r in rel], color=BLUE, marker="o", lw=2)
    style(axes[1], f"Calibration (ECE {summary['ece']:.2f})", "mean grader score in bin", "true success rate in bin")
    sw = summary["threshold_sweep"]
    axes[2].plot([r["threshold"] for r in sw], [r["false_positive_rate"] for r in sw], color=ORANGE, lw=2, label="false-positive rate")
    axes[2].plot([r["threshold"] for r in sw], [r["recall"] for r in sw], color=BLUE, lw=2, label="recall")
    axes[2].plot([r["threshold"] for r in sw], [r["balanced_accuracy"] for r in sw], color=INK, lw=2, label="balanced accuracy")
    style(axes[2], f"Threshold sweep (AUROC {summary['auroc'] if summary['auroc'] is None else round(summary['auroc'], 2)})", "threshold", "rate")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scores_calibration.png", dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    figs.append("scores_calibration.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    per = summary["overall"].get("per_task", {})
    ks = sorted(per, key=int)
    axes[0].bar([int(k) - 0.2 for k in ks], [per[k]["fp"] / max(per[k]["n"], 1) for k in ks], 0.4, color=ORANGE, label="false positives")
    axes[0].bar([int(k) + 0.2 for k in ks], [per[k]["fn"] / max(per[k]["n"], 1) for k in ks], 0.4, color=BLUE, label="false negatives")
    style(axes[0], "Grader errors per task (share of the task's episodes)", "LIBERO-10 task id", "share")
    axes[0].legend(frameon=False, fontsize=8)
    bg = summary["by_goals_satisfied"]
    if bg:
        keys = list(bg)
        axes[1].bar(range(len(keys)), [bg[k]["grader_label_rate"] for k in keys], color=ORANGE, width=0.5)
        for i, k in enumerate(keys):
            axes[1].text(i, bg[k]["grader_label_rate"] + 0.02, f"n={bg[k]['n']}", ha="center", color=INK2, fontsize=8)
        axes[1].set_xticks(range(len(keys)), keys)
        axes[1].set_ylim(0, 1.1)
    style(axes[1], "Grader says 'success' vs goals actually satisfied", "goal predicates satisfied at the end", "grader success rate")
    fig.tight_layout()
    fig.savefig(out / "tasks_partial.png", dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    figs.append("tasks_partial.png")

    # ---------------- gallery ----------------
    gal = out / "gallery"
    gal.mkdir(exist_ok=True)
    idx_fp = [i for i in np.argsort(-score) if oracle[i] == 0 and label[i] == 1][: a.gallery]
    idx_fn = [i for i in np.argsort(score) if oracle[i] == 1 and label[i] == 0][: a.gallery]
    idx_tp = [i for i in np.argsort(-score) if oracle[i] == 1 and label[i] == 1][: max(2, a.gallery // 4)]

    def card(i, kind):
        e = eps[i]
        strip = gal / f"{e['id']}_frames.png"
        if not strip.exists():
            iio.imwrite(strip, frame_strip(e["agentview"]))
        curves = gal / f"{e['id']}_curves.png"
        ex = e.get("grader_extra", {})
        if not curves.exists() and ex.get("progress"):
            fig, ax = plt.subplots(figsize=(4, 2.2))
            t = np.linspace(0, 1, len(ex["progress"]))
            ax.plot(t, ex["progress"], color=BLUE, lw=2, label="progress")
            ax.plot(t, ex["success_probs"], color=ORANGE, lw=2, label="success prob.")
            ax.set_ylim(-0.02, 1.02)
            style(ax, "", "episode time", "")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            fig.savefig(curves, dpi=110, facecolor="#fcfcfb")
            plt.close(fig)
        g = goals.get(e["id"])
        goal_txt = ""
        if g:
            goal_txt = "<br>".join(
                f"{'✔' if pr['satisfied'] else '✘'} {html.escape(' '.join(pr['predicate']))}"
                + (f" (distance {pr['distance']:.3f} m)" if pr.get("distance") is not None else "")
                for pr in g["predicates"])
        rel_video = pathlib.Path(e["agentview"]).resolve()
        return (f"<div class='card {kind}'><div class='meta'><b>{kind}</b> · task {e['task_id']}: "
                f"{html.escape(e.get('task', '') or '')} · score {score[i]:.2f} · oracle: "
                f"{'success' if oracle[i] else 'failure'}{' (succeeded earlier, undone)' if undo[i] else ''} · "
                f"iteration {e['iteration']}</div><img src='gallery/{strip.name}' class='strip'>"
                + (f"<img src='gallery/{curves.name}' class='curves'>" if curves.exists() else "")
                + f"<div class='goals'>{goal_txt}</div><a href='file://{rel_video}'>video</a></div>")

    tasks_text = {}
    for e in eps:
        if "task" in e:
            tasks_text[e["task_id"]] = e["task"]
    s = summary["overall"]
    body = [f"<h1>Grader analysis: {html.escape(a.run.name)}</h1>",
            f"<p>{len(eps)} training episodes. Grader said success for {s['grader_success_rate']:.1%}, "
            f"oracle for {s['oracle_success_rate']:.1%}. Accuracy {s['accuracy']:.1%}, false-positive rate "
            f"{(s['false_positive_rate'] or 0):.1%}, false-negative rate {(s['false_negative_rate'] or 0):.1%}, "
            f"AUROC {summary['auroc'] if summary['auroc'] is None else round(summary['auroc'], 3)}.</p>"]
    body += [f"<img src='{f}' class='fig'>" for f in figs]
    for title, idx, kind in (("Most confident false positives (failures the grader rewarded)", idx_fp, "FP"),
                             ("Most confident false negatives (successes the grader missed)", idx_fn, "FN"),
                             ("Reference: confident true positives", idx_tp, "TP")):
        body.append(f"<h2>{title}</h2>")
        body += [card(i, kind) for i in idx]
    page = ("<!doctype html><meta charset='utf-8'><title>Grader analysis</title><style>"
            "body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1400px;margin:auto;padding:16px}"
            ".fig{width:100%;margin:8px 0}.card{border:1px solid #e4e3df;border-radius:8px;padding:10px;margin:10px 0}"
            ".FP{border-left:5px solid #eb6834}.FN{border-left:5px solid #2a78d6}.TP{border-left:5px solid #1baf7a}"
            ".strip{width:100%}.curves{height:160px}.meta{margin-bottom:6px}.goals{font-size:13px;color:#52514e;margin:6px 0}"
            "</style>" + "\n".join(body))
    (out / "gallery.html").write_text(page)
    print(json.dumps({k: summary[k] for k in ("episodes", "auroc", "ece", "success_then_undo", "by_goals_satisfied")}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
