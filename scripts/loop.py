"""The self-improvement loop: roll out, keep what the judge approves, fine-tune, take the exam, repeat.

    python3 scripts/loop.py --config configs/rft/middle_bowl.json --judge sim [--audit-with robometer]   # control
    python3 scripts/loop.py --config configs/rft/middle_bowl.json --judge robometer                      # experiment

Pure orchestration (standard library only); the work runs in subprocesses with the right interpreters:
the policy server and the fine-tune in openpi's environment, the simulator in LIBERO's, the grader in Robometer's.

Directory layout, under runs/<run_name>/<judge>/:
    round_0/exam/                     the base checkpoint's exam; metrics.json has round 0
    round_k/rollouts/                 attempts by the round k-1 checkpoint (seeded random layouts, videos, frames)
    round_k/kept.json                 the judge's selection; the only place a verdict becomes training data
    round_k/lerobot/rft/train         LeRobot dataset of the kept episodes (and rft/all when log_full_dataset)
    round_k/checkpoint/               the fine-tuned checkpoint of round k
    round_k/exam/                     100 attempts of the round k checkpoint on the fixed exam seeds
    round_k/metrics.json              round, n_rollouts, n_kept, true_success, judged_success, seed, task, ...

Every stage is resumable: re-running the command skips stages whose outputs exist. Non-negotiable 1 is enforced in
rft/select.py: with --judge robometer the simulator's flag is logged on every episode and never read for selection;
with --judge sim and --audit-with robometer the grader scores everything and never selects.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rft import metrics, select, sidecar  # noqa: E402
from rft.config import LoopConfig  # noqa: E402

OPENPI = REPO / "third_party" / "openpi"
OPENPI_PY = OPENPI / ".venv" / "bin" / "python"


def log(msg: str) -> None:
    print(f"[loop {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ policy server


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class PolicyServer:
    """openpi's stock serve_policy.py, run with the venv interpreter, on a checkpoint that holds params/ and assets/."""

    def __init__(self, cfg: LoopConfig, checkpoint: pathlib.Path, log_path: pathlib.Path):
        self.cfg, self.checkpoint, self.log_path, self.proc = cfg, checkpoint.resolve(), log_path, None

    def __enter__(self):
        if _port_open(self.cfg.port):
            raise RuntimeError(f"port {self.cfg.port} is already in use; stop the other server or change the config")
        cmd = [str(OPENPI_PY), "scripts/serve_policy.py", "--port", str(self.cfg.port), "--env", "LIBERO",
               "policy:checkpoint", "--policy.config", self.cfg.policy_config, "--policy.dir", str(self.checkpoint)]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a")
        self._log.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        self._log.flush()
        self.proc = subprocess.Popen(cmd, cwd=str(OPENPI), stdout=self._log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while not _port_open(self.cfg.port):
            if self.proc.poll() is not None:
                raise RuntimeError(f"policy server exited with {self.proc.returncode}; see {self.log_path}")
            if time.time() - t0 > 900:
                self.__exit__(None, None, None)
                raise RuntimeError(f"policy server did not open port {self.cfg.port} in 15 min; see {self.log_path}")
            time.sleep(2)
        log(f"policy server up on {self.cfg.port} ({self.checkpoint.name}) after {time.time() - t0:.0f}s")
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()
        log("policy server stopped")


# ------------------------------------------------------------------------------------------------ stages


def _run(cmd: list[str], log_path: pathlib.Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
        f.flush()
        subprocess.run([str(c) for c in cmd], cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, check=True)


def rollouts_done(out: pathlib.Path, seeds: list[int]) -> bool:
    ids = {r["id"] for r in sidecar.read_records(out / sidecar.FILENAME)}
    return all(f"ep_{s:07d}" in ids for s in seeds) and (out / "metrics.json").exists()


def run_rollouts(cfg: LoopConfig, args, out: pathlib.Path, seeds_kind: str, round_index: int | None,
                 checkpoint: pathlib.Path, lerobot_root: pathlib.Path | None) -> None:
    seeds = cfg.rollout_seeds(round_index) if seeds_kind == "rollout" else cfg.exam_seeds()
    if rollouts_done(out, seeds):
        log(f"{out.relative_to(REPO)}: already complete")
        return
    cmd = [OPENPI_PY, REPO / "scripts" / "rollout_and_log.py", "--config", args.config, "--out", out,
           "--seeds", seeds_kind, "--policy-checkpoint", checkpoint]
    if round_index is not None:
        cmd += ["--round", round_index]
    if lerobot_root is not None:
        cmd += ["--lerobot", lerobot_root, "--repo-id", "rft/all"]
    if args.no_video:
        cmd.append("--no-video")
    with PolicyServer(cfg, checkpoint, out.parent / "server.log"):
        log(f"{seeds_kind}: {len(seeds)} episodes -> {out.relative_to(REPO)}")
        _run(cmd, out.parent / f"{seeds_kind}.log")


def grade(cfg: LoopConfig, rollout_dir: pathlib.Path) -> None:
    from grader import robometer  # noqa: PLC0415

    if robometer.all_graded(rollout_dir):
        log(f"{rollout_dir.relative_to(REPO)}: already graded")
        return
    log(f"grading {rollout_dir.relative_to(REPO)} with {cfg.grader.name}")
    robometer.grade_rollouts(rollout_dir, cfg.grader)


def select_kept(cfg: LoopConfig, judge: str, round_dir: pathlib.Path) -> dict:
    kept_path = round_dir / "kept.json"
    if kept_path.exists():
        return json.loads(kept_path.read_text())
    records = [r for r in sidecar.read_records(round_dir / "rollouts" / sidecar.FILENAME) if not r.get("error")]
    sim = judge == select.SIM_JUDGE
    # --judge says which kind of judge; the config names the grader that wrote the scores (general or in-domain
    # checkpoint), and select.label refuses scores written by any other grader.
    grader_name = None if sim else cfg.grader.name
    threshold = None if sim else cfg.grader.threshold
    ids = select.select_episodes(records, judge if sim else grader_name, threshold)
    kept = {"judge": judge, "grader_name": grader_name, "threshold": threshold, "n_rollouts": len(records),
            "n_kept": len(ids), "ids": ids, "min_kept": cfg.min_kept, "enough": len(ids) >= cfg.min_kept}
    kept_path.write_text(json.dumps(kept, indent=2))
    return kept


def pack_train(round_dir: pathlib.Path, kept: dict) -> pathlib.Path:
    from rft import lerobot_pack  # noqa: PLC0415  (only for is_packed; the packing runs in openpi's env)

    root = round_dir / "lerobot" / "rft" / "train"
    if lerobot_pack.is_packed(root):
        log("train dataset already packed")
        return root
    ids_path = round_dir / "kept_ids.json"
    ids_path.write_text(json.dumps(kept["ids"]))
    _run([OPENPI_PY, "-m", "rft.lerobot_pack", round_dir / "rollouts", root, "rft/train", "--ids", ids_path],
         round_dir / "pack.log")
    return root


def finetune(cfg: LoopConfig, args, round_dir: pathlib.Path, init_checkpoint: pathlib.Path) -> pathlib.Path:
    out = round_dir / "checkpoint"
    info = out / "train_info.json"
    if info.exists():
        final = pathlib.Path(json.loads(info.read_text())["final_checkpoint"])
        if (final / "params").exists():
            log(f"checkpoint already trained: {final.relative_to(REPO)}")
            return final
    log(f"fine-tuning from {init_checkpoint.relative_to(REPO)}")
    cmd = [OPENPI_PY, REPO / "scripts" / "finetune_round.py", "--config", args.config,
           "--dataset-home", round_dir / "lerobot", "--repo-id", "rft/train",
           "--init-checkpoint", init_checkpoint, "--out", out, "--exp-name", round_dir.name]
    if args.train_steps:
        cmd += ["--steps", args.train_steps]
    _run(cmd, round_dir / "train.log")
    return pathlib.Path(json.loads(info.read_text())["final_checkpoint"])


def exam_metrics(exam_dir: pathlib.Path, judge: str, graded: bool, cfg: LoopConfig) -> tuple[dict, dict | None]:
    records = [r for r in sidecar.read_records(exam_dir / sidecar.FILENAME) if not r.get("error")]
    k = sum(1 for r in records if r["sim_success"])
    true = metrics.rate_with_ci(k, len(records))
    judged = None
    if graded:
        scores = [r for r in records if r.get("grader_score") is not None]
        kj = sum(1 for r in scores if float(r["grader_score"]) >= cfg.grader.threshold)
        judged = metrics.rate_with_ci(kj, len(scores))
        judged["agreement"] = metrics.agreement(
            [int(r["sim_success"]) for r in scores], [int(float(r["grader_score"]) >= cfg.grader.threshold) for r in scores],
            [float(r["grader_score"]) for r in scores])
    return true, judged


def rollout_stats(round_dir: pathlib.Path, cfg: LoopConfig, graded: bool) -> dict:
    records = [r for r in sidecar.read_records(round_dir / "rollouts" / sidecar.FILENAME) if not r.get("error")]
    k = sum(1 for r in records if r["sim_success"])
    out = {"true_success": metrics.rate_with_ci(k, len(records))}
    if graded:
        scores = [r for r in records if r.get("grader_score") is not None]
        labels = [int(float(r["grader_score"]) >= cfg.grader.threshold) for r in scores]
        out["judged_rate"] = metrics.rate_with_ci(sum(labels), len(labels))
        out["agreement"] = metrics.agreement([int(r["sim_success"]) for r in scores], labels, [float(r["grader_score"]) for r in scores])
    return out


# ------------------------------------------------------------------------------------------------ rounds


def round_zero(cfg: LoopConfig, args, run_dir: pathlib.Path, graded: bool) -> None:
    round_dir = run_dir / "round_0"
    metrics_path = round_dir / "metrics.json"
    if metrics_path.exists():
        return
    base = REPO / cfg.base_checkpoint
    exam_dir = round_dir / "exam"
    run_rollouts(cfg, args, exam_dir, "exam", None, base, None)
    if graded:
        grade(cfg, exam_dir)
    true, judged = exam_metrics(exam_dir, args.judge, graded, cfg)
    m = {"round": 0, "n_rollouts": 0, "n_kept": 0, "true_success": true, "judged_success": judged,
         "seed": {"exam_seeds": [cfg.exam_seeds()[0], cfg.exam_seeds()[-1]]}, "task": cfg.task_name,
         "task_id": cfg.task_id, "suite": cfg.suite, "judge": args.judge, "threshold": cfg.grader.threshold,
         "checkpoint": str(base.relative_to(REPO)), "status": "ok"}
    metrics_path.write_text(json.dumps(m, indent=2))
    log(f"round 0: true {true['rate']:.3f} [{true['ci95'][0]:.2f}, {true['ci95'][1]:.2f}]"
        + (f", judged {judged['rate']:.3f}" if judged else ""))


def previous_checkpoint(cfg: LoopConfig, run_dir: pathlib.Path, k: int) -> pathlib.Path:
    if k == 1 or not cfg.continue_from_previous:
        return REPO / cfg.base_checkpoint
    info = run_dir / f"round_{k - 1}" / "checkpoint" / "train_info.json"
    return pathlib.Path(json.loads(info.read_text())["final_checkpoint"])


def run_round(cfg: LoopConfig, args, run_dir: pathlib.Path, k: int, graded: bool) -> bool:
    """Returns False when the loop must stop (too few kept episodes)."""
    round_dir = run_dir / f"round_{k}"
    metrics_path = round_dir / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        log(f"round {k}: already done ({m['status']})")
        return m["status"] == "ok"
    t_round = time.time()
    init_ckpt = previous_checkpoint(cfg, run_dir, k)
    rollout_dir = round_dir / "rollouts"
    timings = {}

    t = time.time()
    run_rollouts(cfg, args, rollout_dir, "rollout", k, init_ckpt,
                 (round_dir / "lerobot" / "rft" / "all") if cfg.log_full_dataset else None)
    timings["rollouts_s"] = time.time() - t

    if graded:
        t = time.time()
        grade(cfg, rollout_dir)
        timings["grade_s"] = time.time() - t

    kept = select_kept(cfg, args.judge, round_dir)
    log(f"round {k}: judge {args.judge} kept {kept['n_kept']} of {kept['n_rollouts']} (min {cfg.min_kept})")
    base_metrics = {"round": k, "n_rollouts": kept["n_rollouts"], "n_kept": kept["n_kept"], "task": cfg.task_name,
                    "task_id": cfg.task_id, "suite": cfg.suite, "judge": args.judge, "threshold": kept["threshold"],
                    "grader_name": cfg.grader.name if graded else None,
                    "seed": {"rollout_seeds": [cfg.rollout_seeds(k)[0], cfg.rollout_seeds(k)[-1]],
                             "exam_seeds": [cfg.exam_seeds()[0], cfg.exam_seeds()[-1]], "train_seed": cfg.train.seed},
                    "init_checkpoint": str(init_ckpt), "rollouts": rollout_stats(round_dir, cfg, graded)}
    if not kept["enough"]:
        m = {**base_metrics, "true_success": None, "judged_success": None, "status": "stopped_min_kept", "timings": timings}
        metrics_path.write_text(json.dumps(m, indent=2))
        log(f"round {k}: fewer than {cfg.min_kept} kept; stopping here and reporting it")
        return False

    t = time.time()
    pack_train(round_dir, kept)
    timings["pack_s"] = time.time() - t
    t = time.time()
    final = finetune(cfg, args, round_dir, init_ckpt)
    timings["train_s"] = time.time() - t

    t = time.time()
    exam_dir = round_dir / "exam"
    run_rollouts(cfg, args, exam_dir, "exam", None, final, None)
    if graded:
        grade(cfg, exam_dir)
    timings["exam_s"] = time.time() - t

    true, judged = exam_metrics(exam_dir, args.judge, graded, cfg)
    m = {**base_metrics, "true_success": true, "judged_success": judged, "checkpoint": str(final),
         "status": "ok", "timings": {**timings, "round_s": time.time() - t_round}}
    metrics_path.write_text(json.dumps(m, indent=2))
    log(f"round {k}: true {true['rate']:.3f} [{true['ci95'][0]:.2f}, {true['ci95'][1]:.2f}]"
        + (f", judged {judged['rate']:.3f}" if judged else "") + f", {timings.get('round_s', 0) / 3600:.1f} h")
    return True


def final_practice(cfg: LoopConfig, args, run_dir: pathlib.Path, rounds: int, graded: bool) -> None:
    """Practice attempts by the last checkpoint, on the seeds one more round would use. Evaluation only: nothing is
    selected or trained on. Every earlier checkpoint gets this second sample from the next round's rollouts."""
    k = rounds + 1
    info = run_dir / f"round_{rounds}" / "checkpoint" / "train_info.json"
    checkpoint = pathlib.Path(json.loads(info.read_text())["final_checkpoint"])
    out = run_dir / f"round_{k}" / "rollouts"
    run_rollouts(cfg, args, out, "rollout", k, checkpoint, None)
    if graded:
        grade(cfg, out)
    log(f"final practice: {cfg.rollouts_per_round} attempts by the round {rounds} checkpoint -> {out.relative_to(REPO)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=pathlib.Path)
    p.add_argument("--judge", required=True, choices=["sim", "robometer"])
    p.add_argument("--audit-with", choices=["robometer"], default=None,
                   help="with --judge sim: grade every episode for the audit without ever using it to select")
    p.add_argument("--rounds", type=int, default=None, help="override the config")
    p.add_argument("--run-root", type=pathlib.Path, default=REPO / "runs")
    p.add_argument("--no-video", action="store_true", help="skip mp4s (not possible when a grader is involved)")
    p.add_argument("--train-steps", type=int, default=None, help="override training steps (smoke tests)")
    p.add_argument("--port", type=int, default=None,
                   help="override the config's policy-server port; loops running at once each need their own")
    p.add_argument("--final-practice", action="store_true",
                   help="after the last round, rollouts_per_round more attempts by the final checkpoint (not trained on)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.config = args.config.resolve()

    cfg = LoopConfig.load(args.config)
    if args.port is not None:
        cfg.port = args.port
    rounds = args.rounds or cfg.rounds
    if args.final_practice and cfg.rollout_seeds(rounds + 1)[-1] >= cfg.exam_seed_base:
        p.error("the final practice set's seeds would reach the exam seeds; lower rollout_seed_base")
    graded = args.judge != select.SIM_JUDGE or args.audit_with is not None
    if graded and args.no_video:
        p.error("grading needs the videos; drop --no-video")
    if args.judge != select.SIM_JUDGE and args.audit_with:
        p.error("--audit-with is for the control loop; the experiment loop grades anyway")
    run_dir = args.run_root / cfg.run_name / args.judge
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    if args.port is not None:
        # The stages are subprocesses that read the config file themselves, so the override reaches them as a file.
        args.config = run_dir / "config.json"
    log(f"run {run_dir.relative_to(REPO)}: judge {args.judge}, grader {'on' if graded else 'off'}, {rounds} rounds, "
        f"task {cfg.task_id} {cfg.task_name!r}")
    if args.dry_run:
        for k in range(1, rounds + 1):
            log(f"  round {k}: seeds {cfg.rollout_seeds(k)[0]}..{cfg.rollout_seeds(k)[-1]}, exam {cfg.exam_seeds()[0]}..{cfg.exam_seeds()[-1]}")
        return
    round_zero(cfg, args, run_dir, graded)
    completed = all(run_round(cfg, args, run_dir, k, graded) for k in range(1, rounds + 1))
    if completed and args.final_practice:
        final_practice(cfg, args, run_dir, rounds, graded)
    log("done")


if __name__ == "__main__":
    main()
