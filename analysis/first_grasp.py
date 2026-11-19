"""Which bowl each attempt grabs first, read from the logged gripper path. No vision, no GPU.

Two steps, because frames.npz is large and stays on the training box:

    # on the box, LIBERO client env (numpy): one small grasp_trace.json per rollout folder
    third_party/openpi/examples/libero/.venv/bin/python analysis/first_grasp.py dump runs/middle_bowl
    # anywhere, pure Python: a table per set of attempts and one CSV row per episode
    python3 analysis/first_grasp.py count runs/middle_bowl --config configs/rft/middle_bowl.json \
        --csv results/rft_middle_bowl/episodes.csv

A grab is a stretch of at least MIN_LEN trace samples with the gripper commanded closed. Its place is the gripper
position at the first closed sample, assigned to the nearest known spot. Exams are logged without frames, so only
practice attempts (round_k/rollouts) have traces; exam rows in the CSV leave the trace columns empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rft.metrics import wilson  # noqa: E402

STRIDE = 4  # trace keeps every 4th control step
MIN_LEN = 2  # samples; shorter closures are command blips
MIN_LIFT = 0.04  # metres the gripper must rise during a grab for it to count as carrying something
PLATE = "plate"

# Gripper x, y (world metres) at the first closed sample, clustered over the 3,941 grabs of the six practice sets of
# runs/middle_bowl (k-means, k=4). Specific to LIBERO-90 KITCHEN_SCENE2: three identical bowls and a plate. Names
# follow the task language; in the upright camera view the middle bowl is the leftmost and the back bowl the topmost.
KITCHEN_SCENE2_SPOTS: dict[str, tuple[float, float]] = {
    "back": (-0.147, 0.059),
    "middle": (-0.080, 0.135),
    "front": (0.043, 0.111),
    PLATE: (-0.010, -0.031),
}


def nearest_spot(point: list[float], spots: dict[str, tuple[float, float]]) -> str:
    return min(spots, key=lambda name: math.hypot(point[0] - spots[name][0], point[1] - spots[name][1]))


def grabs(trace: dict, min_len: int = MIN_LEN) -> list[dict]:
    """Stretches of closed-gripper command: where each started, where it ended, and how far the gripper rose."""
    cmd, eef = trace["cmd"], trace["eef"]
    out, i, n = [], 0, len(cmd)
    while i < n:
        if cmd[i] <= 0:
            i += 1
            continue
        j = i
        while j < n and cmd[j] > 0:
            j += 1
        if j - i >= min_len:
            out.append({
                "start": i,
                "end": j,
                "pick": eef[i],
                "release": eef[min(j, n - 1)],
                "released": j < n,
                "lift": max(p[2] for p in eef[i:j]) - eef[i][2],
            })
        i = j
    return out


def describe(trace: dict, spots: dict[str, tuple[float, float]]) -> dict:
    gs = grabs(trace)
    sequence = [nearest_spot(g["pick"], spots) for g in gs]
    delivered = [
        s for s, g in zip(sequence, gs)
        if g["released"] and g["lift"] > MIN_LIFT and nearest_spot(g["release"], spots) == PLATE
    ]
    return {
        "first_grab": sequence[0] if sequence else "none",
        "n_grabs": len(gs),
        "sequence": sequence,
        "delivered": delivered,
    }


def one_wrong_delivery(d: dict, target: str) -> bool:
    """Exactly one bowl carried to the plate, it is not the target, and no third bowl was grabbed."""
    if len(d["delivered"]) != 1:
        return False
    bowl = d["delivered"][0]
    if bowl in (target, PLATE):
        return False
    return all(s in (bowl, target, PLATE) for s in d["sequence"])


def dump(run_dir: pathlib.Path, force: bool) -> None:
    import numpy as np  # noqa: PLC0415  (only the box has it)

    for folder in sorted(run_dir.glob("*/round_*/rollouts")):
        out_path = folder / "grasp_trace.json"
        if out_path.exists() and not force:
            print(f"{folder}: grasp_trace.json exists, skipping (--force to redo)")
            continue
        out = {}
        for ep in sorted(folder.glob("ep_*")):
            frames = ep / "frames.npz"
            if not frames.exists():
                continue
            z = np.load(frames)
            state, actions = z["state"], z["actions"]
            out[ep.name] = {
                "eef": np.round(state[::STRIDE, :3], 4).tolist(),
                "finger": np.round(state[::STRIDE, 6], 4).tolist(),
                "cmd": np.round(actions[::STRIDE, 6], 2).tolist(),
            }
        out_path.write_text(json.dumps(out))
        print(f"{folder}: {len(out)} episodes")


def load_set(folder: pathlib.Path, threshold: float, spots: dict, target: str) -> list[dict]:
    records = [json.loads(line) for line in (folder / "episodes_extra.jsonl").read_text().splitlines() if line.strip()]
    trace_path = folder / "grasp_trace.json"
    traces = json.loads(trace_path.read_text()) if trace_path.exists() else {}
    extra = {p.name.split(".")[1]: json.loads(p.read_text()) for p in sorted(folder.glob("grader_out.*.json"))
             if not p.name.endswith(".manifest.json")}
    rows = []
    for r in records:
        score = r.get("grader_score")
        row = {
            "id": r["id"],
            "seed": r["seed"],
            "sim_success": bool(r["sim_success"]),
            "n_steps": r["n_steps"],
            "grader_score": score,
            "approved": None if score is None else score >= threshold,
            "first_grab": None,
            "n_grabs": None,
            "one_wrong_delivery": None,
        }
        if r["id"] in traces:
            d = describe(traces[r["id"]], spots)
            row.update(first_grab=d["first_grab"], n_grabs=d["n_grabs"], one_wrong_delivery=one_wrong_delivery(d, target))
        for tag, scores in extra.items():
            row[f"score_{tag}"] = scores.get(r["id"], {}).get("score")
        rows.append(row)
    return rows


def collect(run_dir: pathlib.Path, threshold: float, spots: dict, target: str) -> list[dict]:
    """Every logged episode under run_dir, tagged with the loop, the kind of set, and the checkpoint that made it."""
    rows = []
    for loop_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        rounds = sorted(loop_dir.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))
        for round_dir in rounds:
            k = int(round_dir.name.split("_")[1])
            for kind, sub, checkpoint in (("practice", "rollouts", k - 1), ("exam", "exam", k)):
                folder = round_dir / sub
                if not (folder / "episodes_extra.jsonl").exists():
                    continue
                for row in load_set(folder, threshold, spots, target):
                    rows.append({"loop": loop_dir.name, "set": kind, "checkpoint": checkpoint, **row})
    return rows


def _rate(k: int, n: int) -> str:
    return f"{k:3d}/{n:3d} {k / n:.2f}" if n else "  0/  0  n/a"


def print_tables(rows: list[dict], target: str) -> None:
    practice = [r for r in rows if r["set"] == "practice" and r["first_grab"] is not None]
    clean = [r for r in practice if r["sim_success"] and r["n_grabs"] == 1]
    on_target = sum(r["first_grab"] == target for r in clean)
    print(f"check: {on_target} of {len(clean)} one-grab successes start at the {target} bowl")
    print()
    print(f"{'loop':<10} ckpt   n  true [95%]          first={target:<7} success if so  success if not  "
          "approved  failures approved  one wrong delivery")
    for loop, ckpt in sorted({(r["loop"], r["checkpoint"]) for r in practice}):
        s = [r for r in practice if r["loop"] == loop and r["checkpoint"] == ckpt]
        n, wins = len(s), sum(r["sim_success"] for r in s)
        lo, hi = wilson(wins, n)
        first = [r for r in s if r["first_grab"] == target]
        other = [r for r in s if r["first_grab"] != target]
        fails = [r for r in s if not r["sim_success"]]
        print(f"{loop:<10} {ckpt:>4} {n:>3}  {wins / n:.2f} [{lo:.2f}, {hi:.2f}]  {_rate(len(first), n)}   "
              f"{_rate(sum(r['sim_success'] for r in first), len(first))}    "
              f"{_rate(sum(r['sim_success'] for r in other), len(other))}     "
              f"{sum(bool(r['approved']) for r in s) / n:.2f}      {_rate(sum(bool(r['approved']) for r in fails), len(fails))}"
              f"        {sum(bool(r['one_wrong_delivery']) for r in fails):3d}")
    print()
    print("exam and practice of the same checkpoint (fresh layouts both, different seeds):")
    for loop, ckpt in sorted({(r["loop"], r["checkpoint"]) for r in rows}):
        parts = []
        total_k = total_n = 0
        for kind in ("exam", "practice"):
            s = [r for r in rows if r["loop"] == loop and r["checkpoint"] == ckpt and r["set"] == kind]
            if s:
                k = sum(r["sim_success"] for r in s)
                parts.append(f"{kind} {_rate(k, len(s))}")
                total_k, total_n = total_k + k, total_n + len(s)
        lo, hi = wilson(total_k, total_n)
        print(f"  {loop:<10} ckpt {ckpt}: {'  '.join(parts)}  both {_rate(total_k, total_n)} [{lo:.2f}, {hi:.2f}]")


def write_csv(rows: list[dict], path: pathlib.Path) -> None:
    fields = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else int(r[k]) if isinstance(r[k], bool) else r[k]) for k in fields})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="write grasp_trace.json next to each rollout folder's episodes (needs numpy)")
    d.add_argument("run_dir", type=pathlib.Path)
    d.add_argument("--force", action="store_true")
    c = sub.add_parser("count", help="tables and a per-episode CSV from the traces and sidecars")
    c.add_argument("run_dir", type=pathlib.Path)
    c.add_argument("--config", required=True, type=pathlib.Path, help="loop config; supplies the grader threshold")
    c.add_argument("--target", default="middle", help="the spot the instruction names")
    c.add_argument("--csv", type=pathlib.Path, default=None)
    args = p.parse_args()
    if args.command == "dump":
        dump(args.run_dir, args.force)
        return
    threshold = json.loads(args.config.read_text())["grader"]["threshold"]
    rows = collect(args.run_dir, threshold, KITCHEN_SCENE2_SPOTS, args.target)
    print_tables(rows, args.target)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
