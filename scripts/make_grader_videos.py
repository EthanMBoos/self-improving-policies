"""Write grader-oriented videos from frames.npz for rollouts that were logged before the worker produced them.

Runs in the LIBERO client environment (Python 3.8), which has numpy and imageio with ffmpeg.

Robometer's LIBERO training frames are the raw agentview render flipped vertically (its dataset loader does
frames[:, ::-1, :, :]). The policy, openpi, and our logger use the raw render rotated 180 degrees ("upright"), which
is the left-right mirror of that. So from the upright frames in frames.npz:
    agentview.mp4         upright, what the policy sees (the worker writes this)
    agentview_grader.mp4  Robometer's training orientation = upright mirrored left-right (the worker writes this too, since 2026-11-04)
    agentview_raw.mp4     the raw render, upside down; what Robometer's own online wrapper feeds it (--raw)

    third_party/openpi/examples/libero/.venv/bin/python scripts/make_grader_videos.py runs/smoke/sim/round_1/rollouts --raw
"""

from __future__ import annotations

import argparse
import pathlib

import imageio
import numpy as np

VIDEO_FPS = 20


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollout_dir", type=pathlib.Path)
    p.add_argument("--raw", action="store_true", help="also write agentview_raw.mp4")
    p.add_argument("--force", action="store_true", help="rewrite videos that exist")
    args = p.parse_args()
    n = 0
    for npz_path in sorted(args.rollout_dir.glob("ep_*/frames.npz")):
        ep_dir = npz_path.parent
        targets = {"agentview_grader.mp4": lambda f: f[:, :, ::-1, :]}
        if args.raw:
            targets["agentview_raw.mp4"] = lambda f: f[:, ::-1, ::-1, :]
        todo = {name: fn for name, fn in targets.items() if args.force or not (ep_dir / name).exists()}
        if not todo:
            continue
        frames = np.load(npz_path)["image"]
        for name, fn in todo.items():
            imageio.mimwrite(ep_dir / name, np.ascontiguousarray(fn(frames)), fps=VIDEO_FPS)
        n += 1
    print("episodes written: %d" % n)


if __name__ == "__main__":
    main()
