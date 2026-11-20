"""Time LIBERO env creation and rendering for one task. Prints steps/sec, or FAIL.

  python egl_probe.py [suite] [task_id]
"""
import fcntl, os, sys, time, pathlib
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial"
tid = int(sys.argv[2]) if len(sys.argv) > 2 else 0
task = benchmark.get_benchmark_dict()[suite]().get_task(tid)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
tag = "%s/%d" % (suite, tid)
try:
    t0 = time.time()
    with open("/tmp/egl-%s.lock" % os.environ.get("SLURM_JOB_ID", "local"), "w") as lock:  # same lock as rollouts.py
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.reset()
    t = time.time()
    for _ in range(20):
        env.step([0.0] * 6 + [-1.0])
    print("PROBE", tag, "create %.1fs, %.1f steps/s" % (t - t0, 20 / (time.time() - t)), flush=True)
except Exception as e:
    print("PROBE", tag, "FAIL", repr(e)[:200], flush=True)
