"""Part 2, Phase A: which grader + selection rule keeps a clean training set? Offline, no training.

  python phase_a.py --pool $EXP/pools/exp1_train --calib-pool $EXP/pools/main \
      --scores general=<train scores> libero=<train scores> \
      --calib-scores general=<main scores> libero=<main scores> \
      --features <train features_siglip.npz> --calib-features <main features_siglip.npz> [--out $EXP/exp1/sets]

Graders: Robometer general and LIBERO fine-tuned (score files), and a linear probe on frozen features trained on
labelled episodes from the calibration pool (all of them, or a random 100), then applied to the training pool.
Rules: keep the top 25% / 50% of episodes globally or within each task (no labels needed), and for score files the
balanced and strict thresholds picked on the calibration pool's validation split (as in exp1_select.py).
For each: episodes kept, tasks covered, failures kept, rho (share of failures in the kept set = 1 - precision),
successes kept. With --out, writes <grader>_<rule>.txt names files for the selections, ready for fine-tuning.
"""
import argparse
import collections
import json
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze import is_val, thresholds


def metas(pool):
    out = {}
    for p in (pathlib.Path(pool) / "episodes").iterdir():
        if (p / "meta.json").exists():
            m = json.loads((p / "meta.json").read_text())
            out[m["name"]] = m
    return out


def read_scores(path, keep):
    s = {}
    for l in open(path):
        r = json.loads(l)
        if r["name"] in keep and r["score"] is not None:
            s[r["name"]] = r["score"]
    return s


def feats(path):
    f = np.load(path)
    parts = [f["agent_first"], f["agent_last"], f["wrist_first"], f["wrist_last"]] + ([f["text"]] if "text" in f else [])
    return list(f["names"]), np.concatenate(parts, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--calib-pool", required=True)
    ap.add_argument("--scores", nargs="*", default=[], help="name=path, scores on the training pool")
    ap.add_argument("--calib-scores", nargs="*", default=[], help="name=path, same graders on the calibration pool")
    ap.add_argument("--features")
    ap.add_argument("--calib-features")
    ap.add_argument("--strict-fpr", type=float, default=0.10)
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    tm, cm = metas(a.pool), metas(a.calib_pool)
    graders, calib = {}, {}
    for kv in a.scores:
        k, p = kv.split("=", 1)
        graders[k] = read_scores(p, tm)
    for kv in a.calib_scores:
        k, p = kv.split("=", 1)
        calib[k] = read_scores(p, cm)
    if a.features and a.calib_features:
        tn, Xt = feats(a.features)
        cn, Xc = feats(a.calib_features)
        yc = np.array([cm[n]["success"] for n in cn])
        for label, idx in (("probe_all", np.arange(len(cn))), ("probe_100", rng.choice(len(cn), 100, replace=False))):
            sc = StandardScaler().fit(Xc[idx])
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xc[idx]), yc[idx])
            graders[label] = dict(zip(tn, clf.predict_proba(sc.transform(Xt))[:, 1]))

    names = sorted(set(tm).intersection(*[set(g) for g in graders.values()]))
    y = {n: bool(tm[n]["success"]) for n in names}
    task = {n: tm[n]["task_id"] for n in names}
    ns = sum(y.values())
    print("training pool: %d episodes, %d successes, %d failures, %d tasks" % (
        len(names), ns, len(names) - ns, len(set(task.values()))))

    def top(s, frac, per_task):
        groups = collections.defaultdict(list)
        for n in names:
            groups[task[n] if per_task else 0].append(n)
        kept = []
        for g in groups.values():
            g = sorted(g, key=lambda n: -s[n])
            kept += g[: int(round(frac * len(g)))]
        return kept

    rows = []
    for gname, s in graders.items():
        rules = {"global50": top(s, .5, False), "global25": top(s, .25, False),
                 "task50": top(s, .5, True), "task25": top(s, .25, True)}
        if gname in calib:
            c = calib[gname]
            val = [n for n in c if is_val(n, 0.3)]
            thr, _ = thresholds(np.array([cm[n]["success"] for n in val]), np.array([c[n] for n in val]))
            vf = np.sort([c[n] for n in val if not cm[n]["success"]])
            strict = vf[min(len(vf) - 1, int(np.floor((1 - a.strict_fpr) * len(vf))))] + 1e-9
            rules["balanced"] = [n for n in names if s[n] >= thr]
            rules["strict"] = [n for n in names if s[n] >= strict]
        for rname, kept in rules.items():
            nf = sum(not y[n] for n in kept)
            rows.append((gname, rname, len(kept), len({task[n] for n in kept}), nf, nf / max(len(kept), 1),
                         len(kept) - nf, kept))

    print("\n%-10s %-9s %6s %6s %9s %6s %10s" % ("grader", "rule", "kept", "tasks", "failures", "rho", "successes"))
    print("%-10s %-9s %6d %6d %9d %6.2f %10d" % ("oracle", "clean", ns, len({task[n] for n in names if y[n]}), 0, 0, ns))
    for g, r, k, t, nf, rho, nsk, _ in sorted(rows, key=lambda r: r[5]):
        print("%-10s %-9s %6d %6d %9d %6.2f %10d" % (g, r, k, t, nf, rho, nsk))
    if a.out:
        out = pathlib.Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for g, r, *_, kept in rows:
            (out / ("%s_%s.txt" % (g, r))).write_text("\n".join(sorted(kept)) + "\n")


if __name__ == "__main__":
    main()
