"""Turn a list of pool episodes into a LeRobot dataset in openpi's LIBERO format, for Experiment 1 fine-tuning.

  python make_dataset.py --pool $EXP/pools/main --names keep.txt --repo-id exp1/clean   (openpi env)

Matches examples/libero/convert_libero_data_to_lerobot.py: 256x256 images rotated 180 degrees from raw robosuite
frames (what the policy sees, not the true view the graders get), 8-d state, 7-d actions, task text as the prompt,
fps 10. Writes to $HF_LEROBOT_HOME/<repo-id>, replacing any existing copy, and prints the dataset's size,
number of tasks and how many of its episodes actually failed (for the noisy rows).
"""
import argparse
import collections
import json
import pathlib
import shutil

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--names", required=True, help="file with one episode name per line")
    ap.add_argument("--repo-id", required=True)
    a = ap.parse_args()
    names = pathlib.Path(a.names).read_text().split()
    out = HF_LEROBOT_HOME / a.repo_id
    if out.exists():
        shutil.rmtree(out)
    img = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
    ds = LeRobotDataset.create(
        repo_id=a.repo_id, robot_type="panda", fps=10, root=out,
        features={"image": img, "wrist_image": img,
                  "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                  "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}},
        image_writer_threads=10, image_writer_processes=5)
    tasks, failed, frames = collections.Counter(), 0, 0
    for n in names:
        ep = pathlib.Path(a.pool) / "episodes" / n
        m = json.loads((ep / "meta.json").read_text())
        with np.load(ep / "episode.npz") as z:  # read each array once: indexing the NpzFile decompresses it every time
            agent, wrist = z["agentview"][:, ::-1, ::-1], z["wrist"][:, ::-1, ::-1]
            state, actions = z["state"].astype(np.float32), z["actions"].astype(np.float32)
        for t in range(len(actions)):
            ds.add_frame({"image": np.ascontiguousarray(agent[t]), "wrist_image": np.ascontiguousarray(wrist[t]),
                          "state": state[t], "actions": actions[t], "task": m["task"]})
        ds.save_episode()
        tasks[m["task_id"]] += 1
        failed += not m["success"]
        frames += len(actions)
    print("wrote %s: %d episodes, %d frames, %d tasks, %d failed episodes (%.0f%%)" % (
        out, len(names), frames, len(tasks), failed, 100 * failed / max(len(names), 1)))


if __name__ == "__main__":
    main()
