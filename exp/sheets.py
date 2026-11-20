"""Contact sheets for checking episodes by eye.

  python sheets.py --pool $EXP/pools/main --out sheets/ [--names file] [--frames 6]

One PNG per episode: a row of evenly spaced third-person frames and a row of wrist frames, in the true camera view
(raw[::-1]), with the task and LIBERO's label in the title. Also writes orientation.png: one raw frame, the true
view and the mirrored (policy) view side by side.
"""
import argparse
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()
    pool, out = pathlib.Path(a.pool), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = sorted(p for p in (pool / "episodes").iterdir() if (p / "meta.json").exists())
    if a.names:
        keep = set(pathlib.Path(a.names).read_text().split())
        eps = [p for p in eps if p.name in keep]
    for k, ep in enumerate(eps):
        meta = json.loads((ep / "meta.json").read_text())
        d = np.load(ep / "episode.npz")
        if k == 0:
            raw = d["agentview"][-1]
            row = np.concatenate([raw, raw[::-1], raw[::-1, ::-1]], axis=1)
            im = Image.fromarray(row).resize((row.shape[1] * 2, row.shape[0] * 2))
            label(im, "raw  |  true view raw[::-1]  |  policy view raw[::-1,::-1]").save(out / "orientation.png")
        idx = np.linspace(0, len(d["agentview"]) - 1, a.frames).round().astype(int)
        top = np.concatenate(list(d["agentview"][idx][:, ::-1]), axis=1)
        bot = np.concatenate(list(d["wrist"][idx][:, ::-1]), axis=1)
        im = Image.fromarray(np.concatenate([np.zeros((16, top.shape[1], 3), np.uint8), top, bot], axis=0))
        label(im, "%s | %s | %s | %d steps" % ("SUCCESS" if meta["success"] else "FAIL", meta["name"], meta["task"],
                                               meta["steps"])).save(out / (ep.name + ".png"))


if __name__ == "__main__":
    main()
