"""Offline goal analysis of recorded RL episodes (RLinf env, which has LIBERO).

For every episode with a saved final simulator state, restore that state in a LIBERO env of the same task and
evaluate each goal predicate of the task separately (e.g. "put both moka pots on the stove" has two), plus the
distance between the objects each binary predicate relates (e.g. moka pot -> stove). Also re-runs LIBERO's
success check on the restored state, which must equal the oracle label logged during training.

Usage: LIBERO_CONFIG_PATH=... third_party/RLinf/.venv/bin/python scripts/rl/goal_report.py outputs/rl/runs/<run>
Writes <run>/goal_report.jsonl (one line per episode).
"""

import json
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    run = pathlib.Path(sys.argv[1])
    out_path = run / "goal_report.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()}
    eps = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    eps = [e for e in eps if e.get("final_state") and e["id"] not in done]
    if not eps:
        print("nothing to do")
        return

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    by_task = {}
    for e in eps:
        by_task.setdefault(e["task_id"], []).append(e)
    n_mismatch = 0
    with open(out_path, "a") as f:
        for task_id, task_eps in sorted(by_task.items()):
            task = suite.get_task(task_id)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
            env.seed(0)
            env.reset()
            inner = env.env
            goal = inner.parsed_problem["goal_state"]
            for e in task_eps:
                env.set_init_state(np.load(e["final_state"]))
                preds = []
                for g in goal:
                    entry = {"predicate": list(g), "satisfied": bool(inner._eval_predicate(g)), "distance": None}
                    if len(g) == 3:
                        try:
                            a = np.asarray(inner.object_states_dict[g[1]].get_geom_state()["pos"])
                            b = np.asarray(inner.object_states_dict[g[2]].get_geom_state()["pos"])
                            entry["distance"] = float(np.linalg.norm(a - b))
                            entry["distance_xy"] = float(np.linalg.norm(a[:2] - b[:2]))
                        except Exception:  # noqa: BLE001
                            pass
                    preds.append(entry)
                success = bool(inner._check_success())
                if success != bool(e["oracle_success_at_end"]):
                    n_mismatch += 1
                f.write(json.dumps({
                    "id": e["id"], "task_id": task_id, "predicates": preds,
                    "n_satisfied": sum(p["satisfied"] for p in preds), "n_goals": len(preds),
                    "restored_success": success, "oracle_success_at_end": bool(e["oracle_success_at_end"]),
                }) + "\n")
            env.close()
            print(f"task {task_id}: {len(task_eps)} episodes, goals: {[g[0] for g in goal]}", flush=True)
    print(f"done; restored-state success disagrees with logged oracle for {n_mismatch} episodes")


if __name__ == "__main__":
    main()
