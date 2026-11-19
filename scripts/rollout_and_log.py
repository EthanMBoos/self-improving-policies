"""Roll out one task against a running policy server, log every episode, and pack them for training.

Build-order step 2 and the exam harness in one tool. Runs in the openpi environment and launches the LIBERO client
(Python 3.8) as a subprocess, because the simulator and the LeRobot library cannot share an interpreter; the packing
happens in this same invocation, so there is no separate conversion step.

    third_party/openpi/.venv/bin/python scripts/rollout_and_log.py --config configs/rft/middle_bowl.json \
        --out runs/x/round_1/rollouts --seeds rollout --round 1 --policy-checkpoint checkpoints/pi05_libero \
        [--lerobot runs/x/round_1/lerobot/rft/all] [--no-video]

    # the exam: fixed seeds, no frames, no dataset
    third_party/openpi/.venv/bin/python scripts/rollout_and_log.py --config ... --out runs/x/round_1/exam \
        --seeds exam --policy-checkpoint <dir>

Outputs: <out>/episodes_extra.jsonl, <out>/<id>/{frames.npz,agentview.mp4,wrist.mp4}, <out>/metrics.json
(true success with a 95% Wilson interval), and the LeRobot dataset at --lerobot if given.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rft import metrics, sidecar  # noqa: E402
from rft.config import LoopConfig  # noqa: E402

OPENPI = REPO / "third_party" / "openpi"
CLIENT_PY = OPENPI / "examples" / "libero" / ".venv" / "bin" / "python"


def worker_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{OPENPI / 'third_party' / 'libero'}".lstrip(":")
    env.setdefault("MUJOCO_GL", "egl")
    env["LIBERO_CONFIG_PATH"] = str(REPO / "third_party" / "libero_config")
    return env


def run_worker(cfg: LoopConfig, out: pathlib.Path, seeds: list[int], *, round_index, policy_checkpoint,
               video: bool, frames: bool, host: str = "0.0.0.0") -> None:
    cmd = [
        str(CLIENT_PY), str(REPO / "scripts" / "rollout_worker.py"),
        "--suite", cfg.suite, "--task-id", str(cfg.task_id), "--out", str(out),
        "--host", host, "--port", str(cfg.port),
        "--resize", str(cfg.resize), "--replan-steps", str(cfg.replan_steps), "--wait-steps", str(cfg.wait_steps),
        "--policy-checkpoint", str(policy_checkpoint), "--seeds", *map(str, seeds),
    ]
    if round_index is not None:
        cmd += ["--round", str(round_index)]
    if not video:
        cmd.append("--no-video")
    if not frames:
        cmd.append("--no-frames")
    log = out / "worker.log"
    out.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        subprocess.run(cmd, cwd=str(OPENPI), env=worker_env(), stdout=f, stderr=subprocess.STDOUT, check=True)


def write_metrics(out: pathlib.Path, cfg: LoopConfig, seeds: list[int], kind: str) -> dict:
    records = sidecar.read_records(out / sidecar.FILENAME)
    wanted = {f"ep_{s:07d}" for s in seeds}
    recs = [r for r in records if r["id"] in wanted]
    ok = [r for r in recs if not r.get("error")]
    k = sum(1 for r in ok if r["sim_success"])
    m = {
        "kind": kind,
        "task": cfg.task_name or (recs[0]["task"] if recs else None),
        "suite": cfg.suite,
        "task_id": cfg.task_id,
        "n_requested": len(seeds),
        "n_logged": len(recs),
        "n_errors": len(recs) - len(ok),
        "true_success": metrics.rate_with_ci(k, len(ok)),
        "mean_steps": (sum(r["n_steps"] for r in ok) / len(ok)) if ok else None,
        "mean_steps_success": (sum(r["n_steps"] for r in ok if r["sim_success"]) / k) if k else None,
        "seeds": [seeds[0], seeds[-1]] if seeds else [],
    }
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--seeds", required=True, help="'rollout' (needs --round), 'exam', or a comma list of ints")
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--policy-checkpoint", default="", help="label stored with each episode")
    p.add_argument("--lerobot", type=pathlib.Path, default=None, help="also pack all episodes into a dataset here")
    p.add_argument("--repo-id", default="rft/all")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    cfg = LoopConfig.load(args.config)
    if args.seeds == "rollout":
        if args.round is None:
            p.error("--seeds rollout needs --round")
        seeds, kind, frames = cfg.rollout_seeds(args.round), "rollout", True
    elif args.seeds == "exam":
        seeds, kind, frames = cfg.exam_seeds(), "exam", False
    else:
        seeds, kind, frames = [int(s) for s in args.seeds.split(",")], "custom", True

    t0 = time.time()
    run_worker(cfg, args.out, seeds, round_index=args.round, policy_checkpoint=args.policy_checkpoint,
               video=not args.no_video, frames=frames, host=args.host)
    m = write_metrics(args.out, cfg, seeds, kind)
    m["rollout_s"] = time.time() - t0
    print(json.dumps({k: v for k, v in m.items() if k != "seeds"}, indent=2))
    if args.lerobot is not None:
        from rft import lerobot_pack  # noqa: PLC0415

        m["pack"] = lerobot_pack.pack(args.out, args.lerobot, args.repo_id)
        print(json.dumps(m["pack"], indent=2))
    # written last, so a crash in packing leaves no metrics.json and the loop re-runs this stage on resume
    (args.out / "metrics.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
