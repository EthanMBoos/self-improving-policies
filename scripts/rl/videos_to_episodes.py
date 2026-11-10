"""Build an episodes.jsonl from labelled rollout videos, so scripts/rl/grade_offline.py can grade them like an RL run.

openpi's LIBERO eval client keeps one video per task and outcome, named rollout_<task words>_{success,failure}.mp4,
in the upright view the policy sees (raw render rotated 180 degrees), the same view as the RL worker's videos. The
outcome in the file name is LIBERO's success check. Each episode records which directory it came from as "source".
Paths are written absolute, so run this on the machine that will grade.

    python3 scripts/rl/videos_to_episodes.py outputs/rl/committed results/libero_eval/*/videos_* \
        results/libero90_sweep_50/videos_candidates
"""

import argparse
import json
import pathlib
import re

NAME = re.compile(r"^rollout_(?P<task>.+)_(?P<outcome>success|failure)\.mp4$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=pathlib.Path, help="written: <out_dir>/episodes.jsonl")
    p.add_argument("video_dirs", nargs="+", type=pathlib.Path)
    a = p.parse_args()

    eps = []
    for d in a.video_dirs:
        d = d.resolve()
        source = f"{d.parent.name}/{d.name}"
        for v in sorted(d.glob("rollout_*.mp4")):
            m = NAME.match(v.name)
            if not m:
                continue
            ok = m["outcome"] == "success"
            eps.append({"id": f"{source}/{v.stem}", "source": source, "task": m["task"].replace("_", " "),
                        "oracle_success_at_end": ok, "oracle_success_once": ok, "agentview": str(v),
                        "wrist": None, "final_state": None, "grader_label": None, "grader_score": None})
    if not eps:
        raise SystemExit("no rollout_*_{success,failure}.mp4 videos found")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    n_ok = sum(e["oracle_success_at_end"] for e in eps)
    print(f"wrote {a.out_dir / 'episodes.jsonl'}: {len(eps)} episodes, {n_ok} successes, {len(eps) - n_ok} failures")


if __name__ == "__main__":
    main()
