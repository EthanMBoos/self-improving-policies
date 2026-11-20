"""Add fixture joint positions (drawers, doors, knobs) to scene.json of episodes recorded before rollouts.py saved them.

Replays each episode's saved actions from its init state without rendering, records the joints at the first policy
step and at the end, and checks that the replay reproduces the recorded success. Runs in the LIBERO client env.

  python replay_joints.py --pool $EXP/pools/main [--shard 0 --shards 8]
"""
import argparse
import json
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from rollouts import DUMMY_ACTION, fixture_joints

WAIT = 10  # rollouts.py num_steps_wait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    eps = sorted(p for p in (pathlib.Path(a.pool) / "episodes").iterdir() if (p / "meta.json").exists())
    eps = [p for k, p in enumerate(eps) if k % a.shards == a.shard]
    envs = {}
    mismatch = 0
    for ep in eps:
        scene = json.loads((ep / "scene.json").read_text())
        if "joints" in scene.get("end", {}):
            continue
        m = json.loads((ep / "meta.json").read_text())
        suite = benchmark.get_benchmark_dict()[m["suite"]]()
        key = (m["suite"], m["task_id"])
        if key not in envs:
            for e in envs.values():
                e.close()
            task = suite.get_task(m["task_id"])
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            envs = {key: ControlEnv(bddl_file_name=str(bddl), has_renderer=False, has_offscreen_renderer=False,
                                    use_camera_obs=False)}
            envs[key].seed(m["seed"])
        env = envs[key]
        env.reset()
        env.set_init_state(suite.get_task_init_states(m["task_id"])[m["init"]])
        for _ in range(WAIT):
            env.step(DUMMY_ACTION)
        start = fixture_joints(env.env.sim)
        done = False
        for act in np.load(ep / "episode.npz")["actions"]:
            _, _, done, _ = env.step(act.tolist())
            if done:
                break
        end = fixture_joints(env.env.sim)
        ok = bool(done) == bool(m["success"])
        mismatch += not ok
        scene["start"]["joints"], scene["end"]["joints"] = start, end
        scene["replay_matches_success"] = ok
        (ep / "scene.json").write_text(json.dumps(scene))
        moved = {k: round(end[k] - start[k], 3) for k in start if abs(end[k] - start[k]) > 0.01}
        print(ep.name, "success=%s replay=%s" % (m["success"], bool(done)), "moved:", moved, flush=True)
    print("done, %d replay mismatches" % mismatch)


if __name__ == "__main__":
    main()
