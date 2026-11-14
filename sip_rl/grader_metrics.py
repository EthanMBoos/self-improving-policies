"""Grader-vs-oracle agreement metrics (numpy only; used during RL and in the offline grader benchmark)."""

from __future__ import annotations

import collections

import numpy as np


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks of ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        idx = np.where(allv == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & ((scores < hi) if hi < 1 else (scores <= hi))
        if m.any():
            ece += m.mean() * abs(scores[m].mean() - labels[m].mean())
    return float(ece)


def agreement(oracle: np.ndarray, label: np.ndarray, score: np.ndarray | None = None, tasks=None) -> dict:
    """Binary grader labels (and optional scores in [0, 1]) against oracle success, plus per-task error rates."""
    oracle = np.asarray(oracle).astype(int)
    label = np.asarray(label).astype(int)
    tp = int(((label == 1) & (oracle == 1)).sum())
    fp = int(((label == 1) & (oracle == 0)).sum())
    fn = int(((label == 0) & (oracle == 1)).sum())
    tn = int(((label == 0) & (oracle == 0)).sum())
    n = len(oracle)
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    out = {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": _safe_div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "balanced_accuracy": (recall + tnr) / 2 if recall is not None and tnr is not None else None,
        "oracle_success_rate": _safe_div(int(oracle.sum()), n),
        "grader_success_rate": _safe_div(int(label.sum()), n),
    }
    # The gap the policy is optimized against: positive = grader is more optimistic than reality.
    if n:
        out["success_rate_gap"] = out["grader_success_rate"] - out["oracle_success_rate"]
    if score is not None:
        score = np.clip(np.asarray(score, dtype=float), 0, 1)
        out["auroc"] = auroc(score, oracle)
        out["brier"] = float(np.mean((score - oracle) ** 2)) if n else None
        out["ece"] = expected_calibration_error(score, oracle) if n else None
        out["score_mean_success"] = float(score[oracle == 1].mean()) if (oracle == 1).any() else None
        out["score_mean_failure"] = float(score[oracle == 0].mean()) if (oracle == 0).any() else None
    if tasks is not None:
        per = collections.defaultdict(lambda: [0, 0, 0, 0])  # n, errors, fp, fn
        for t, o, l in zip(tasks, oracle, label):
            per[t][0] += 1
            per[t][1] += int(o != l)
            per[t][2] += int(l == 1 and o == 0)
            per[t][3] += int(l == 0 and o == 1)
        out["per_task"] = {str(t): {"n": v[0], "error_rate": v[1] / v[0], "fp": v[2], "fn": v[3]} for t, v in per.items()}
    return out
