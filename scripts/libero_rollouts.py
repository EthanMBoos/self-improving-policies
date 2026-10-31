"""Roll out a served policy in LIBERO and log every episode (Python 3.8 LIBERO client environment).

Based on openpi's examples/libero/main.py (same preprocessing, seed handling, replanning and step limits), but stores
per-episode data for analysis and for auditing reward models (e.g. VLM graders) against the simulator's success flag:

    <out>/<suite>/episodes.jsonl                     one line per episode (task, init state, success, steps, ...)
    <out>/<suite>/<task_id>_<init>/agentview.mp4     third-person camera, upright, 256x256, 20 fps (control rate)
    <out>/<suite>/<task_id>_<init>/wrist.mp4         wrist camera
    <out>/<suite>/<task_id>_<init>/data.npz          actions [T,7], states [T,8] (eef pos, axis-angle, gripper)

Usage (see scripts/eval_checkpoints.sh, which sets PYTHONPATH / MUJOCO_GL / LIBERO_CONFIG_PATH):
    python scripts/libero_rollouts.py --suite libero_spatial --trials-per-task 5 --out outputs/eval/x
"""

import argparse
import collections
import json
import logging
import math
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=sorted(MAX_STEPS))
    p.add_argument("--trials-per-task", type=int, default=5, help="uses LIBERO's fixed init states 0..N-1")
    p.add_argument("--tasks", type=int, nargs="*", help="task ids (default: all)")
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--tag", default="", help="free-form label stored with each episode (e.g. checkpoint step)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    out = args.out / args.suite
    out.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    log_path = out / "episodes.jsonl"
    if log_path.exists():  # resume: skip episodes already logged
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            done_keys.add((r["task_id"], r["init_state"]))
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    max_steps = MAX_STEPS[args.suite]

    for task_id in args.tasks if args.tasks else range(suite.n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RESOLUTION, camera_widths=ENV_RESOLUTION)
        env.seed(args.seed)  # seed affects object positions even with a fixed init state (as in openpi)

        for init_idx in range(args.trials_per_task):
            if (task_id, init_idx) in done_keys:
                continue
            env.reset()
            obs = env.set_init_state(init_states[init_idx])
            plan = collections.deque()
            frames, wrist_frames, actions, states, infer_ms = [], [], [], [], []
            t, success, error = 0, False, None
            t0 = time.time()
            while t < max_steps + args.wait_steps:
                try:
                    if t < args.wait_steps:  # let objects settle
                        obs, _, done, _ = env.step(DUMMY_ACTION)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # rotate 180 (= upright)
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    state = np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    )
                    frames.append(img)
                    wrist_frames.append(wrist)
                    if not plan:
                        element = {
                            "observation/image": image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(img, args.resize, args.resize)
                            ),
                            "observation/wrist_image": image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist, args.resize, args.resize)
                            ),
                            "observation/state": state,
                            "prompt": str(task.language),
                        }
                        ti = time.time()
                        chunk = client.infer(element)["actions"]
                        infer_ms.append(1000 * (time.time() - ti))
                        plan.extend(chunk[: args.replan_steps])
                    action = np.asarray(plan.popleft())
                    actions.append(action)
                    states.append(state)
                    obs, _, done, _ = env.step(action.tolist())
                    t += 1
                    if done:
                        success = True
                        break
                except Exception as e:  # noqa: BLE001
                    error = repr(e)
                    logging.error(f"episode error: {error}")
                    break

            ep_dir = out / f"{task_id:02d}_{init_idx:02d}"
            ep_dir.mkdir(exist_ok=True)
            if not args.no_video and frames:
                imageio.mimwrite(ep_dir / "agentview.mp4", frames, fps=20)
                imageio.mimwrite(ep_dir / "wrist.mp4", wrist_frames, fps=20)
            np.savez_compressed(ep_dir / "data.npz", actions=np.asarray(actions), states=np.asarray(states))
            record = {
                "suite": args.suite,
                "task_id": task_id,
                "task": task.language,
                "init_state": init_idx,
                "success": success,
                "steps": len(actions),
                "max_steps": max_steps,
                "inference_calls": len(infer_ms),
                "infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
                "wall_s": time.time() - t0,
                "seed": args.seed,
                "tag": args.tag,
                "error": error,
                "dir": ep_dir.name,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logging.info(f"[{args.suite}] task {task_id} init {init_idx}: {'SUCCESS' if success else 'fail'} "
                         f"({len(actions)} steps)")
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
