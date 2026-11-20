"""Exp 2: how many labelled episodes does a linear probe on frozen features need to match the big graders?

  python probe.py --pool $EXP/pools/main --features $EXP/pools/main/features_siglip.npz [--scores scores/*.jsonl]

Two test splits, both by whole episode:
  episodes  the same held-out test episodes analyze.py uses (hash of the name), tasks seen in training
  tasks     a fixed 30% of tasks never seen in training (hash of the task id)
Training sets are random subsets of the remaining episodes, 20 draws per size. Reports mean AUROC and
FPR at the threshold giving 90% recall on the test set (an optimistic operating point, the same for every method).
Feature sets: full (first+last frames, both cameras, text), last-frame only, and full with shuffled labels.
Graders given with --scores are scored on the same test episodes for comparison.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze import auroc, is_val


def fpr_at_recall(y, s, r=0.9):
    pos = np.sort(s[y])
    if not len(pos) or y.all():
        return np.nan
    t = pos[int(np.floor((1 - r) * len(pos)))]
    return (s[~y] >= t).mean()


def task_held_out(task_id, frac=0.3):
    return int(hashlib.md5(("task%d" % task_id).encode()).hexdigest(), 16) % 1000 < frac * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--sizes", default="25,50,100,200,all")
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    f = np.load(a.features)
    names = list(f["names"])
    metas = [json.loads((pathlib.Path(a.pool) / "episodes" / n / "meta.json").read_text()) for n in names]
    y = np.array([m["success"] for m in metas])
    tid = np.array([m["task_id"] for m in metas])
    parts = [f["agent_first"], f["agent_last"], f["wrist_first"], f["wrist_last"]] + ([f["text"]] if "text" in f else [])
    sets = {"full": np.concatenate(parts, 1), "last-frame": np.concatenate([f["agent_last"], f["wrist_last"]], 1)}
    test_ep = np.array([not is_val(n, 0.3) for n in names])  # analyze.py's test split ...
    val_ep = ~test_ep  # ... and its validation split, used here as the pool to draw training sets from
    held = np.array([task_held_out(t) for t in tid])
    splits = {
        "episodes": (val_ep, test_ep),  # train on analyze.py's validation episodes, test on its test episodes
        "tasks": (~held, held),
    }
    rng = np.random.default_rng(0)
    print("%d episodes, %.1f%% success; held-out tasks: %d episodes" % (len(y), 100 * y.mean(), held.sum()))
    for split, (tr_mask, te_mask) in splits.items():
        tr_idx = np.where(tr_mask)[0]
        print("\n== test split: %s  (train pool %d, test %d, test success %.0f%%)" % (
            split, len(tr_idx), te_mask.sum(), 100 * y[te_mask].mean()))
        print("%-22s %6s %8s %8s" % ("features", "n", "AUROC", "FPR@90"))
        for fs, X in list(sets.items()) + [("full, shuffled labels", sets["full"])]:
            for size in a.sizes.split(","):
                n = len(tr_idx) if size == "all" else int(size)
                if n > len(tr_idx):
                    continue
                au, fp = [], []
                for d in range(a.draws if size != "all" else 1):
                    idx = rng.choice(tr_idx, n, replace=False) if size != "all" else tr_idx
                    yt = y[idx].copy()
                    if "shuffled" in fs:
                        rng.shuffle(yt)
                    if yt.all() or (~yt).all():
                        continue
                    sc = StandardScaler().fit(X[idx])
                    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[idx]), yt)
                    s = clf.predict_proba(sc.transform(X[te_mask]))[:, 1]
                    au.append(auroc(y[te_mask], s))
                    fp.append(fpr_at_recall(y[te_mask], s))
                print("%-22s %6d %8.3f %8.3f" % (fs, n, np.nanmean(au), np.nanmean(fp)))
        for path in a.scores:
            rows = {json.loads(l)["name"]: json.loads(l)["score"] for l in open(path)}
            m = te_mask & np.array([n in rows and rows[n] is not None for n in names])
            s = np.array([rows[n] if n in rows and rows[n] is not None else np.nan for n in names])
            print("%-40s %6s %8.3f %8.3f" % (pathlib.Path(path).stem[:40], "-", auroc(y[m], s[m]), fpr_at_recall(y[m], s[m])))


if __name__ == "__main__":
    main()
