"""Roll out a served policy on one LIBERO task with seeded random layouts and log every episode.

Runs in the LIBERO client environment (Python 3.8: third_party/openpi/examples/libero/.venv). Adapted from openpi's
examples/libero/main.py and from scripts/libero_rollouts.py on branch stratos (3db6273): same image preprocessing,
replanning and step limits. Two differences: every episode starts from a fresh seeded random layout instead of one
of LIBERO's 50 shipped initial states (LIBERO's seed() is np.random.seed, and reset() draws object placements from
that generator), and the raw arrays are saved for the LeRobot packer in rft/lerobot_pack.py.

Writes under <out>/:
    episodes_extra.jsonl         one record per episode (keys in rft/sidecar.py); resume skips ids already there
    <id>/frames.npz              image [T,256,256,3] uint8 upright, wrist_image [T,256,256,3], state [T,8], actions [T,7]
    <id>/agentview.mp4, wrist.mp4   20 fps, for the grader and for humans (unless --no-video)

The parent script (scripts/rollout_and_log.py) sets PYTHONPATH, MUJOCO_GL and LIBERO_CONFIG_PATH and runs:
    examples/libero/.venv/bin/python scripts/rollout_worker.py --suite libero_90 --task-id 24 \
        --seeds 1000 1001 1002 --out runs/x/round_1/rollouts --port 8001 --round 1 --policy-checkpoint <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
SIDECAR = "episodes_extra.jsonl"
VIDEO_FPS = 20


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def episode_id(seed):
    return "ep_%07d" % seed


def existing_ids(sidecar_path):
    if not sidecar_path.exists():
        return set()
    return {json.loads(line)["id"] for line in sidecar_path.read_text().splitlines() if line.strip()}


def run_episode(env, client, task_language, max_steps, args, obs):
    """One attempt from the observation `reset` returned. Returns (success, frames, wrist_frames, states, actions, infer_ms, error)."""
    plan = collections.deque()
    frames, wrist_frames, states, actions, infer_ms = [], [], [], [], []
    t, success, error = 0, False, None
    while t < max_steps + args.wait_steps:
        try:
            if t < args.wait_steps:  # let objects settle after the reset
                obs, _, _, _ = env.step(DUMMY_ACTION)
                t += 1
                continue
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # 180 degree rotation = upright
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            state = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ).astype(np.float32)
            if not plan:
                element = {
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize, args.resize)
                    ),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist, args.resize, args.resize)
                    ),
                    "observation/state": state,
                    "prompt": task_language,
                }
                t0 = time.time()
                chunk = client.infer(element)["actions"]
                infer_ms.append(1000.0 * (time.time() - t0))
                plan.extend(chunk[: args.replan_steps])
            action = np.asarray(plan.popleft(), dtype=np.float32)
            frames.append(img)
            wrist_frames.append(wrist)
            states.append(state)
            actions.append(action)
            obs, _, done, _ = env.step(action.tolist())
            t += 1
            if done:
                success = True
                break
        except Exception as e:  # noqa: BLE001
            error = repr(e)
            logging.error("episode error: %s", error)
            break
    return success, frames, wrist_frames, states, actions, infer_ms, error


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=sorted(MAX_STEPS))
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="one episode per seed; id = ep_<seed>")
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None, help="default: the suite's limit")
    p.add_argument("--round", type=int, default=None, help="loop round, stored with each record")
    p.add_argument("--policy-checkpoint", default="", help="label of the checkpoint being served")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-frames", action="store_true", help="skip frames.npz (exam mode: nothing to train on)")
    args = p.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    max_steps = args.max_steps or MAX_STEPS[args.suite]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sidecar_path = out / SIDECAR
    done = existing_ids(sidecar_path)
    todo = [s for s in args.seeds if episode_id(s) not in done]
    logging.info("task %d %r: %d episodes to run, %d already logged", args.task_id, task.language, len(todo), len(done))
    if not todo:
        return

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RESOLUTION, camera_widths=ENV_RESOLUTION)
    try:
        for seed in todo:
            ep_id = episode_id(seed)
            t_start = time.time()
            env.seed(seed)  # np.random.seed under the hood: the next reset draws this seed's object placements
            obs0 = env.reset()
            success, frames, wrist_frames, states, actions, infer_ms, error = run_episode(
                env, client, str(task.language), max_steps, args, obs0
            )
            ep_dir = out / ep_id
            ep_dir.mkdir(exist_ok=True)
            if frames and not args.no_video:
                imageio.mimwrite(ep_dir / "agentview.mp4", frames, fps=VIDEO_FPS)
                imageio.mimwrite(ep_dir / "wrist.mp4", wrist_frames, fps=VIDEO_FPS)
            if frames and not args.no_frames:
                np.savez_compressed(
                    ep_dir / "frames.npz",
                    image=np.stack(frames),
                    wrist_image=np.stack(wrist_frames),
                    state=np.stack(states),
                    actions=np.stack(actions),
                )
            record = {
                "id": ep_id,
                "suite": args.suite,
                "task_id": args.task_id,
                "task": str(task.language),
                "task_name": task.name,
                "seed": seed,
                "init_state_source": "seeded_reset",
                "sim_success": bool(success),
                "n_steps": len(actions),
                "max_steps": max_steps,
                "inference_calls": len(infer_ms),
                "infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
                "wall_s": time.time() - t_start,
                "error": error,
                "dir": ep_id,
                "policy_checkpoint": args.policy_checkpoint,
                "round": args.round,
            }
            with sidecar_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            logging.info("%s seed %d: %s in %d steps", ep_id, seed, "SUCCESS" if success else "fail", len(actions))
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
