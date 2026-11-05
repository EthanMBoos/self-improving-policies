"""Grade the recorded training episodes of a finished RL run with another grader, offline (no RL), so the grader can
be audited against the oracle on on-policy episodes. Run with the RLinf env python (omegaconf).

Usage: third_party/RLinf/.venv/bin/python scripts/rl/grade_offline.py <run_dir> <out_dir> [--grader robometer_soft]
           [--chunk 512] [--batch-size 16] [--path-from OLD --path-to NEW]
Writes <out_dir>/episodes.jsonl in the run format (grader_label / grader_score / grader_extra replaced; reward = what
this grader would have given) and copies goal_report.jsonl, so scripts/rl/analyze_grader.py <out_dir> works on it.
Resumable: graded chunks are kept in <out_dir>/grading/chunk_*/.
"""

import argparse
import json
import os
import pathlib
import shutil
import sys

from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sip_rl.graders import make_grader  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("out_dir", type=pathlib.Path)
    p.add_argument("--grader", default="robometer_soft", help="preset name under sip.graders in the RL config")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16, help="the GPU is not shared here, so larger than in RL")
    p.add_argument("--path-from", default=None, help="rewrite video paths (e.g. after moving the run directory)")
    p.add_argument("--path-to", default=None)
    a = p.parse_args()
    a.run_dir, a.out_dir = a.run_dir.resolve(), a.out_dir.resolve()  # the grader runs in its own working directory

    os.environ.setdefault("SIP_REPO", str(ROOT))
    cfg = OmegaConf.load(ROOT / "configs/rl/libero10_pi05_lora.yaml")
    preset = OmegaConf.to_container(cfg.sip.graders[a.grader], resolve=True)
    if "args" in preset and "batch_size" in preset["args"]:
        preset["args"]["batch_size"] = a.batch_size
    reward_mode = preset.get("reward", "label")
    grader = make_grader(preset, ROOT)

    eps = [json.loads(line) for line in (a.run_dir / "episodes.jsonl").read_text().splitlines()]
    for e in eps:
        for k in ("agentview", "wrist", "final_state"):
            if a.path_from and e.get(k):
                e[k] = e[k].replace(a.path_from, a.path_to)
        e.setdefault("num_frames", None)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for i in range(0, len(eps), a.chunk):
            work = a.out_dir / "grading" / f"chunk_{i // a.chunk:04d}"
            done = work / "labels.json"
            if done.exists():
                results.update(json.loads(done.read_text()))
                continue
            r = grader.grade(eps[i : i + a.chunk], work)
            r.pop("__wall_s__", None)
            timing = r.pop("__timing__", {})
            done.write_text(json.dumps(r))
            results.update(r)
            print(f"graded {min(i + a.chunk, len(eps))}/{len(eps)} {timing}", flush=True)
    finally:
        grader.close()

    with open(a.out_dir / "episodes.jsonl", "w") as f:
        for e in eps:
            r = results[e["id"]]
            e["grader_label"] = int(r["label"])
            e["grader_score"] = r.get("score")
            e["reward"] = float(r["score"] if reward_mode == "score" else r["label"])
            e["grader_extra"] = {k: v for k, v in r.items() if k not in ("label", "score")}
            e.pop("num_frames", None)
            f.write(json.dumps(e) + "\n")
    if (a.run_dir / "goal_report.jsonl").exists():
        shutil.copy(a.run_dir / "goal_report.jsonl", a.out_dir / "goal_report.jsonl")
    (a.out_dir / "source.json").write_text(json.dumps({"run_dir": str(a.run_dir), "grader": a.grader,
                                                        "preset": preset, "episodes": len(eps)}, indent=2))
    print(f"wrote {a.out_dir / 'episodes.jsonl'} ({len(eps)} episodes)")


if __name__ == "__main__":
    main()
