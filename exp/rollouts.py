"""Run a pi0.5 policy server in LIBERO and save everything about each episode.

Runs in openpi's LIBERO client env (Python 3.8). Based on openpi/examples/libero/main.py, with the same
preprocessing, plus per-step saving. One directory per episode:

  episode.npz   agentview, wrist: (T, 256, 256, 3) uint8, raw robosuite orientation (upside down)
                state: (T, 8) eef pos, axis-angle, gripper qpos  (what the policy saw)
                actions: (T, 7) the action executed after that observation
  agentview.mp4, wrist.mp4   the same frames rotated 180 degrees (upright, as the policy sees them)
  scene.json    object positions and goal predicate values at the first policy step and at the end
  meta.json     suite, task id, task text, init state index, seed, success, steps

A summary line per episode is appended to <out>/episodes.jsonl. Existing episodes are skipped, so a job can be rerun.
"""
import collections
import dataclasses
import fcntl
import json
import logging
import math
import os
import pathlib
import time
from typing import List, Optional

import imageio
import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy

DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


@dataclasses.dataclass
class Args:
    out: str
    suite: str = "libero_90"
    tasks: Optional[List[int]] = None  # default: all tasks in the suite
    inits: List[int] = dataclasses.field(default_factory=lambda: [0])  # init state indices (0-49)
    seed: int = 7  # env seed; openpi's eval uses 7
    tag: str = ""  # appended to the episode name, e.g. the checkpoint
    host: str = "127.0.0.1"
    port: int = 8000
    replan_steps: int = 5
    num_steps_wait: int = 10


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def make_env(task, seed, lock_path):
    """Create the env while holding a file lock: several processes creating EGL contexts at once on one GPU hang."""
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        env.seed(seed)
        env.reset()
    return env


def fixture_joints(sim):
    """Positions of hinge and slide joints that are not the robot's: drawers, doors, stove knobs, etc.
    (Movable objects have free joints and are covered by their body positions.)"""
    out = {}
    for j, name in enumerate(sim.model.joint_names):
        if sim.model.jnt_type[j] in (2, 3) and not name.startswith(("robot", "gripper", "mount")):
            out[name] = round(float(sim.data.qpos[sim.model.jnt_qposadr[j]]), 4)
    return out


def scene_state(env):
    """Object positions, fixture joint positions and each goal predicate's truth value, to tell near-misses from
    obvious failures later."""
    out = {}
    try:
        e = env.env
        out["objects"] = {n: e.sim.data.body_xpos[i].round(4).tolist() for n, i in e.obj_body_id.items()}
        out["joints"] = fixture_joints(e.sim)
        out["goal"] = [[" ".join(map(str, g)), bool(e._eval_predicate(g))] for g in e.parsed_problem["goal_state"]]
    except Exception as ex:  # never let bookkeeping kill a rollout
        out["error"] = repr(ex)
    return out


def run_episode(env, init_state, prompt, client, args, max_steps):
    env.reset()
    obs = env.set_init_state(init_state)
    start = None
    t_start = time.time()
    plan = collections.deque()
    agent, wrist, states, actions = [], [], [], []
    done = False
    for t in range(max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(DUMMY_ACTION)
            continue
        if start is None:
            start = scene_state(env)
        state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
        if not plan:
            def prep(x):
                x = np.ascontiguousarray(x[::-1, ::-1])  # rotate 180, matching openpi training preprocessing
                return image_tools.convert_to_uint8(image_tools.resize_with_pad(x, 224, 224))
            chunk = client.infer({
                "observation/image": prep(obs["agentview_image"]),
                "observation/wrist_image": prep(obs["robot0_eye_in_hand_image"]),
                "observation/state": state,
                "prompt": prompt,
            })["actions"]
            plan.extend(chunk[: args.replan_steps])
        if t % 100 == 0:
            logging.info("  step %d (%.1f steps/s)", t, t / max(time.time() - t_start, 1e-6))
        action = np.asarray(plan.popleft(), dtype=np.float32)
        agent.append(obs["agentview_image"].copy())
        wrist.append(obs["robot0_eye_in_hand_image"].copy())
        states.append(state.astype(np.float32))
        actions.append(action)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
    return (bool(done), np.stack(agent), np.stack(wrist), np.stack(states), np.stack(actions),
            dict(start=start, end=scene_state(env)))


def main(args: Args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    tasks = args.tasks if args.tasks is not None else list(range(suite.n_tasks))
    for attempt in range(30):  # the server can be too busy (first JIT compile) to finish a handshake in time
        try:
            client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
            break
        except Exception as ex:
            logging.warning("connect attempt %d failed: %r", attempt, ex)
            time.sleep(10)
    else:
        raise RuntimeError("could not connect to policy server")
    max_steps = MAX_STEPS[args.suite]
    lock_path = "/tmp/egl-%s.lock" % os.environ.get("SLURM_JOB_ID", "local")
    for task_id in tasks:
        task = suite.get_task(task_id)
        inits = suite.get_task_init_states(task_id)
        env = None
        for i in args.inits:
            name = "%s_t%02d_i%02d_s%d%s" % (args.suite, task_id, i, args.seed, "_" + args.tag if args.tag else "")
            ep = out / "episodes" / name
            if (ep / "meta.json").exists():
                continue
            if env is None:
                env = make_env(task, args.seed, lock_path)
            t0 = time.time()
            success, agent, wrist, states, actions, scene = run_episode(env, inits[i], task.language, client, args, max_steps)
            ep.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ep / "episode.npz", agentview=agent, wrist=wrist, state=states, actions=actions)
            imageio.mimwrite(ep / "agentview.mp4", list(agent[:, ::-1, ::-1]), fps=20)
            imageio.mimwrite(ep / "wrist.mp4", list(wrist[:, ::-1, ::-1]), fps=20)
            meta = dict(name=name, suite=args.suite, task_id=task_id, task=task.language, init=i, seed=args.seed,
                        tag=args.tag, success=success, steps=len(actions), max_steps=max_steps,
                        seconds=round(time.time() - t0, 1))
            (ep / "scene.json").write_text(json.dumps(scene))
            (ep / "meta.json").write_text(json.dumps(meta))
            with open(out / "episodes.jsonl", "a") as f:
                f.write(json.dumps(meta) + "\n")
            logging.info("%s success=%s steps=%d %.0fs", name, success, len(actions), meta["seconds"])
        if env is not None:
            with open(lock_path, "w") as lock:  # closing EGL contexts concurrently is as fragile as creating them
                fcntl.flock(lock, fcntl.LOCK_EX)
                env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(Args))
