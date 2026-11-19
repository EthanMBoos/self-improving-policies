"""Success rates with intervals, and grader-versus-simulator agreement. Pure Python so it runs anywhere.

The agreement metrics follow the definitions in Stratos's sip_rl/grader_metrics.py (branch stratos, 3db6273),
rewritten without numpy.
"""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n, as fractions."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate_with_ci(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Probability that a random positive outranks a random negative; ties count half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def expected_calibration_error(scores: list[float], labels: list[int], bins: int = 10) -> float | None:
    n = len(scores)
    if n == 0:
        return None
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if (lo <= s < hi) or (hi == 1.0 and s == 1.0)]
        if idx:
            mean_s = sum(scores[i] for i in idx) / len(idx)
            mean_l = sum(labels[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(mean_s - mean_l)
    return ece


def agreement(sim: list[int], grader: list[int], scores: list[float] | None = None) -> dict:
    """Confusion matrix of grader labels against simulator truth, plus score-based metrics when scores are given."""
    sim = [int(bool(x)) for x in sim]
    grader = [int(bool(x)) for x in grader]
    n = len(sim)
    tp = sum(1 for s, g in zip(sim, grader) if s and g)
    fp = sum(1 for s, g in zip(sim, grader) if not s and g)
    fn = sum(1 for s, g in zip(sim, grader) if s and not g)
    tn = sum(1 for s, g in zip(sim, grader) if not s and not g)
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    tnr = _div(tn, tn + fp)
    out = {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": _div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
        "false_positive_rate": _div(fp, fp + tn),
        "false_negative_rate": _div(fn, fn + tp),
        "balanced_accuracy": (recall + tnr) / 2 if recall is not None and tnr is not None else None,
        "sim_success_rate": _div(sum(sim), n),
        "grader_success_rate": _div(sum(grader), n),
    }
    if n:
        out["success_rate_gap"] = out["grader_success_rate"] - out["sim_success_rate"]
    if scores is not None:
        s = [min(1.0, max(0.0, float(x))) for x in scores]
        out["auroc"] = auroc(s, sim)
        out["brier"] = sum((a - b) ** 2 for a, b in zip(s, sim)) / n if n else None
        out["ece"] = expected_calibration_error(s, sim)
        out["score_mean_success"] = _div(sum(a for a, b in zip(s, sim) if b), sum(sim))
        out["score_mean_failure"] = _div(sum(a for a, b in zip(s, sim) if not b), n - sum(sim))
    return out


def threshold_sweep(sim: list[int], scores: list[float], thresholds: list[float]) -> dict[str, dict]:
    return {str(t): agreement(sim, [int(s >= t) for s in scores], scores) for t in thresholds}
