"""Pack raw rollout episodes (scripts/rollout_worker.py output) into a LeRobot v2.1 dataset openpi can train on.

Runs in the openpi environment (third_party/openpi/.venv), where the pinned lerobot library lives. The schema is
openpi's own, from examples/libero/convert_libero_data_to_lerobot.py: image, wrist_image (256x256 uint8, "image"
dtype so frames are PNGs and no ffmpeg is needed), state (8), actions (7), task per frame, plus one extra column
sim_success (float32, constant within an episode). openpi's repack transform maps only the keys it knows, so the
extra column never reaches the model. Episode-level metadata goes to <root>/meta/episodes_extra.jsonl.

The dataset is written to an explicit root; the trainer points HF_LEROBOT_HOME at the parent so openpi's loader
finds it under <root parent>/<repo_id> (see scripts/finetune_round.py).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import time

from rft import sidecar

FPS = 10  # the metadata value physical-intelligence/libero uses; one frame per control step in both
FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    "sim_success": {"dtype": "float32", "shape": (1,), "names": ["sim_success"]},
}


def pack(
    rollout_dir: pathlib.Path,
    root: pathlib.Path,
    repo_id: str,
    episode_ids: list[str] | None = None,
    image_writer_threads: int = 8,
    image_writer_processes: int = 4,
) -> dict:
    """Write the episodes (all, or the given ids) into a fresh dataset at `root`. Returns a summary dict."""
    import numpy as np  # noqa: PLC0415 (openpi env only; scripts/loop.py imports this module without numpy)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    rollout_dir, root = pathlib.Path(rollout_dir), pathlib.Path(root)
    records = [r for r in sidecar.read_records(rollout_dir / sidecar.FILENAME) if not r.get("error")]
    if episode_ids is not None:
        wanted = set(episode_ids)
        missing = wanted - {r["id"] for r in records}
        if missing:
            raise KeyError(f"episodes not found or errored in {rollout_dir}: {sorted(missing)[:5]}")
        records = [r for r in records if r["id"] in wanted]
    if not records:
        raise ValueError(f"no episodes to pack from {rollout_dir}")

    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=str(root),
        robot_type="panda",
        features=FEATURES,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )
    packed, n_frames = [], 0
    for ep_index, r in enumerate(records):
        npz = np.load(rollout_dir / r["dir"] / "frames.npz")
        image, wrist, state, actions = npz["image"], npz["wrist_image"], npz["state"], npz["actions"]
        assert len(image) == len(wrist) == len(state) == len(actions) > 0, r["id"]
        sim = np.array([1.0 if r["sim_success"] else 0.0], dtype=np.float32)
        for t in range(len(actions)):
            ds.add_frame(
                {
                    "image": np.ascontiguousarray(image[t]),
                    "wrist_image": np.ascontiguousarray(wrist[t]),
                    "state": state[t].astype(np.float32),
                    "actions": actions[t].astype(np.float32),
                    "sim_success": sim,
                    "task": r["task"],
                }
            )
        ds.save_episode()
        n_frames += len(actions)
        packed.append({**r, "episode_index": ep_index})
    if hasattr(ds, "stop_image_writer"):
        ds.stop_image_writer()
    sidecar.write_records(root / "meta" / sidecar.FILENAME, packed)
    summary = {
        "repo_id": repo_id,
        "root": str(root),
        "n_episodes": len(packed),
        "n_frames": n_frames,
        "n_success": sum(1 for r in packed if r["sim_success"]),
        "pack_s": time.time() - t0,
    }
    (root / "meta" / "pack_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def is_packed(root: pathlib.Path) -> bool:
    root = pathlib.Path(root)
    return (root / "meta" / "info.json").exists() and (root / "meta" / "pack_summary.json").exists()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="pack rollouts into a LeRobot dataset (run with openpi's python)")
    p.add_argument("rollout_dir", type=pathlib.Path)
    p.add_argument("root", type=pathlib.Path)
    p.add_argument("repo_id")
    p.add_argument("--ids", type=pathlib.Path, default=None, help="JSON list of episode ids to include")
    args = p.parse_args()
    ids = json.loads(args.ids.read_text()) if args.ids else None
    print(json.dumps(pack(args.rollout_dir, args.root, args.repo_id, ids), indent=2))


if __name__ == "__main__":
    main()
