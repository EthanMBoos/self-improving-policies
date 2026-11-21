"""Experiment 1: decide which training-pool episodes go into each dataset, and report what each dataset contains.

  python exp1_select.py --pool $EXP/pools/exp1_train --scores $EXP/pools/exp1_train/scores/<grader>.jsonl \
      --calib-pool $EXP/pools/main --calib-scores $EXP/pools/main/scores/<grader>.jsonl --out $EXP/exp1/sets

Datasets (one names file each in --out):
  clean      true successes (LIBERO's check)
  fp30       true successes plus a random 30% of failures, relabelled as successes
  grader     whatever the grader passes, at the balanced-accuracy threshold picked on the calibration pool's
             validation split (the same split analyze.py uses), frozen before looking at the training pool
  grader_strict  the same grader at a strict threshold (at most --strict-fpr of calibration failures pass)
  fp10/fp50, fn10/fn30/fn50   the rest of the sweep: 10/50% of failures added, or 10/30/50% of successes dropped
  matched    the same number of successes and failures per task as the grader set, but chosen at random: same
             error rates and task mix, only *which* failures get in differs (structured vs. random mistakes)

Prints, per dataset: episodes, tasks, failures included (share), successes dropped; and the grader's false-positive
and false-negative rates on the training pool.
"""
import argparse
import collections
import json
import pathlib

import numpy as np

from analyze import is_val, thresholds


def load(pool, scores):
    meta = {}
    for p in (pathlib.Path(pool) / "episodes").iterdir():
        if (p / "meta.json").exists():
            m = json.loads((p / "meta.json").read_text())
            meta[m["name"]] = m
    s = {}
    if scores:
        for l in open(scores):
            r = json.loads(l)
            if r["name"] in meta and r["score"] is not None:
                s[r["name"]] = r["score"]
    return meta, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--calib-pool", required=True)
    ap.add_argument("--calib-scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict-fpr", type=float, default=0.10, help="failure pass rate for the grader_strict set")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cm, cs = load(a.calib_pool, a.calib_scores)
    val = [n for n in cs if is_val(n, 0.3)]
    thr, _ = thresholds(np.array([cm[n]["success"] for n in val]), np.array([cs[n] for n in val]))
    print("grader threshold (balanced accuracy on %d calibration-validation episodes): %.4f" % (len(val), thr))
    vf = np.sort([cs[n] for n in val if not cm[n]["success"]])
    strict = vf[min(len(vf) - 1, int(np.floor((1 - a.strict_fpr) * len(vf))))] + 1e-9  # passes <= strict_fpr of them
    print("strict threshold (at most %.0f%% of calibration-validation failures pass): %.4f" % (100 * a.strict_fpr, strict))

    meta, s = load(a.pool, a.scores)
    names = sorted(n for n in meta if n in s)
    missing = len(meta) - len(names)
    y = {n: bool(meta[n]["success"]) for n in names}
    passed = {n: s[n] >= thr for n in names}
    succ = [n for n in names if y[n]]
    fail = [n for n in names if not y[n]]
    fpr = sum(passed[n] for n in fail) / max(len(fail), 1)
    fnr = sum(not passed[n] for n in succ) / max(len(succ), 1)
    print("training pool: %d episodes graded (%d missing scores), %d successes, %d failures" % (
        len(names), missing, len(succ), len(fail)))
    print("grader on training pool: FPR %.2f (passes %d of %d failures), FNR %.2f (drops %d of %d successes)" % (
        fpr, sum(passed[n] for n in fail), len(fail), fnr, sum(not passed[n] for n in succ), len(succ)))

    sets = {
        "clean": succ,
        "fp30": succ + list(rng.choice(fail, int(round(0.3 * len(fail))), replace=False)),
        "grader": [n for n in names if passed[n]],
    }
    by_task = collections.defaultdict(lambda: ([], []))
    for n in names:
        by_task[meta[n]["task_id"]][0 if y[n] else 1].append(n)
    matched = []
    for t, (ts, tf) in by_task.items():  # same per-task counts as the grader set; only *which* episodes is random
        ks, kf = sum(passed[n] for n in ts), sum(passed[n] for n in tf)
        matched += list(rng.choice(ts, ks, replace=False)) + list(rng.choice(tf, kf, replace=False))
    sets["matched"] = matched
    sets["grader_strict"] = [n for n in names if s[n] >= strict]
    sfp = sum(s[n] >= strict for n in fail) / max(len(fail), 1)
    sfn = sum(s[n] < strict for n in succ) / max(len(succ), 1)
    print("strict grader on training pool: FPR %.2f, FNR %.2f" % (sfp, sfn))
    # The sweep, nested so each level only adds more of the same noise: FP 10 < FP 30 < FP 50 share failures (FP 30's
    # are drawn first, above), FN 10 < FN 30 < FN 50 drop successes in one fixed random order.
    fp30_fail = [n for n in sets["fp30"] if not y[n]]
    rest = [n for n in rng.permutation(fail) if n not in set(fp30_fail)]
    fail_order = list(rng.permutation(fp30_fail)) + rest
    for k in (10, 50):
        sets["fp%d" % k] = succ + fail_order[: int(round(k / 100 * len(fail)))]
    succ_order = list(rng.permutation(succ))
    for k in (10, 30, 50):
        dropped = set(succ_order[: int(round(k / 100 * len(succ)))])
        sets["fn%d" % k] = [n for n in succ if n not in dropped]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("\n%-8s %8s %6s %18s %16s" % ("set", "episodes", "tasks", "failures in it", "successes left out"))
    for k, v in sets.items():
        (out / ("%s.txt" % k)).write_text("\n".join(sorted(v)) + "\n")
        nf = sum(not y[n] for n in v)
        print("%-8s %8d %6d %10d (%4.0f%%) %16d" % (k, len(v), len({meta[n]["task_id"] for n in v}), nf,
                                                  100 * nf / max(len(v), 1), len(succ) - sum(y[n] for n in v)))
    print("\nper-task episodes (task: clean / grader / matched):")
    print("  " + "  ".join("%d:%d/%d/%d" % (t, sum(meta[n]["task_id"] == t for n in sets["clean"]),
                                          sum(meta[n]["task_id"] == t for n in sets["grader"]),
                                          sum(meta[n]["task_id"] == t for n in sets["matched"]))
                         for t in sorted(by_task)))


if __name__ == "__main__":
    main()
