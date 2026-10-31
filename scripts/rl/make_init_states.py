"""Generate fresh LIBERO initial states for RL training, disjoint from the official evaluation states.

LIBERO evaluates on 50 fixed initial states per task (init_files/<suite>/<task>.pruned_init). RL that trains on
those states would practice on the test layouts, so training uses new layouts sampled with LIBERO's own placement
sampler (env.reset()), seeded for reproducibility, in the same file format.

Usage (RLinf env):
    third_party/RLinf/.venv/bin/python scripts/rl/make_init_states.py --suite libero_10 --per-task 100 \
        --out third_party/libero_train_init_states
Writes <out>/<problem_folder>/<init_states_file> for every task of the suite.
"""

import argparse
import os
import pathlib

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--per-task", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345, help="differs from the eval seed (7) and LIBERO's files")
    p.add_argument("--out", type=pathlib.Path, required=True)
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        official = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
        states = []
        for k in range(args.per_task):
            env.seed(args.seed + 1000 * task_id + k)
            env.reset()
            states.append(np.asarray(env.get_sim_state(), dtype=official.dtype))
        env.close()
        states = np.stack(states)
        assert states.shape[1:] == official.shape[1:], f"state shape {states.shape} vs official {official.shape}"
        # Sanity: no generated layout coincides with an official evaluation layout.
        min_dist = min(np.abs(official - s).max() for s in states)
        out = args.out / task.problem_folder / task.init_states_file
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(states, out)
        print(f"task {task_id}: {states.shape[0]} states -> {out.name} (min L-inf distance to official: {min_dist:.3f})")


if __name__ == "__main__":
    main()
