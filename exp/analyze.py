"""Compare graders on a pool against LIBERO's success label.

  python analyze.py --pool $EXP/pools/main scores/*.jsonl [--val 0.3]

Each score file is one grader setting (grade.py output). Episodes are split by whole episode into validation and
test with a fixed hash of the name. Thresholds are chosen on validation and frozen; every number except "AUROC
(all)" is on test only.

Failures are tagged from scene.json:
  obvious    no object moved more than 3 cm and no goal predicate became true
  near-miss  anything else (something was moved, or part of the goal holds)
"""
import argparse
import collections
import hashlib
import itertools
import json
import pathlib

import numpy as np


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        ranks[m] = ranks[m].mean()
    n1 = y.sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(auroc(y[i], s[i]))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)


def is_val(name, frac):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 < frac * 1000


def tag_failure(ep_dir):
    try:
        sc = json.loads((ep_dir / "scene.json").read_text())
        a, b = sc["start"]["objects"], sc["end"]["objects"]
        moved = max(np.linalg.norm(np.subtract(b[k], a[k])) for k in a)
        partial = any(v for _, v in sc["end"]["goal"]) and not any(v for _, v in sc["start"]["goal"])
        return "near-miss" if moved > 0.03 or partial else "obvious"
    except Exception:
        return "unknown"


def thresholds(y, s):
    """Threshold maximising balanced accuracy, and the highest threshold keeping >= 90% recall."""
    cands = np.unique(s)
    best, best_t = -1, cands[0]
    for t in cands:
        p = s >= t
        ba = (p[y].mean() + (~p[~y]).mean()) / 2
        if ba > best:
            best, best_t = ba, t
    pos = np.sort(s[y])
    t90 = pos[int(np.floor(0.1 * len(pos)))] if len(pos) else np.nan
    return best_t, t90


def rates(y, p):
    tp, fp = (p & y).sum(), (p & ~y).sum()
    return dict(fpr=fp / max((~y).sum(), 1), recall=tp / max(y.sum(), 1), precision=tp / max(tp + fp, 1),
                passed=p.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--val", type=float, default=0.3)
    ap.add_argument("scores", nargs="+")
    a = ap.parse_args()
    pool = pathlib.Path(a.pool)
    metas = {}
    for p in (pool / "episodes").iterdir():
        if (p / "meta.json").exists():
            m = json.loads((p / "meta.json").read_text())
            m["kind"] = "success" if m["success"] else tag_failure(p)
            metas[m["name"]] = m
    kinds = collections.Counter(m["kind"] for m in metas.values())
    print("pool: %d episodes, %s" % (len(metas), dict(kinds)))

    graders = {}
    for f in a.scores:
        rows = {}
        for l in open(f):
            r = json.loads(l)
            if r["name"] in metas and r["score"] is not None and not np.isnan(r["score"]):
                rows[r["name"]] = r["score"]
        graders[pathlib.Path(f).stem] = rows
    names = sorted(set.intersection(*[set(g) for g in graders.values()]))
    val = np.array([is_val(n, a.val) for n in names])
    y = np.array([metas[n]["success"] for n in names])
    kind = np.array([metas[n]["kind"] for n in names])
    print("graded by all: %d (val %d, test %d), success %.1f%%\n" % (len(names), val.sum(), (~val).sum(), 100 * y.mean()))

    hdr = "%-34s %6s %15s %6s | %s | %s" % ("grader", "AUROC", "95% CI (test)", "NM", "bal-acc thr: FPR rec prec",
                                          "90%-recall thr: FPR rec prec NM-FPR")
    print(hdr)
    print("-" * len(hdr))
    passed = {}
    for g, rows in graders.items():
        s = np.array([rows[n] for n in names])
        te = ~val
        nm = te & ((kind == "near-miss") | y)
        tb, t90 = thresholds(y[val], s[val])
        lo, hi = boot_ci(y[te], s[te])
        r1, r2 = rates(y[te], s[te] >= tb), rates(y[te], s[te] >= t90)
        nm_f = te & (kind == "near-miss")
        nm_fpr = (s[nm_f] >= t90).mean() if nm_f.any() else np.nan
        print("%-34s %6.3f %7.3f-%-7.3f %6.3f | %.2f %.2f %.2f | %.2f %.2f %.2f %.2f" % (
            g, auroc(y[te], s[te]), lo, hi, auroc(y[nm], s[nm]), r1["fpr"], r1["recall"], r1["precision"],
            r2["fpr"], r2["recall"], r2["precision"], nm_fpr))
        passed[g] = s >= tb

    if len(graders) > 1:
        te_f = (~val) & ~y
        print("\nfailures passed at each grader's bal-acc threshold (test, %d failures):" % te_f.sum())
        for g1, g2 in itertools.combinations(graders, 2):
            a1, a2 = passed[g1][te_f], passed[g2][te_f]
            both, either = (a1 & a2).sum(), (a1 | a2).sum()
            print("  %-30s %-30s  passed %3d / %3d, both %3d, jaccard %.2f" % (g1, g2, a1.sum(), a2.sum(), both,
                                                                         both / max(either, 1)))
        allp = np.all([passed[g][te_f] for g in graders], axis=0)
        print("  passed by every grader: %d" % allp.sum())
        for n in np.array(names)[te_f][allp]:
            print("    %s  %s  (%s)" % (n, metas[n]["task"], metas[n]["kind"]))
        vote = np.mean([passed[g] for g in graders], axis=0)
        print("\nmajority vote: %s" % rates(y[~val], vote[~val] > 0.5))
        avg = np.mean([np.argsort(np.argsort([graders[g][n] for n in names])) for g in graders], axis=0)
        print("rank-average AUROC (test): %.3f" % auroc(y[~val], avg[~val]))


if __name__ == "__main__":
    main()
