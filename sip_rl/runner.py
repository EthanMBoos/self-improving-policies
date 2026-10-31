"""RLinf EmbodiedRunner with a grading phase, comparison metrics and a GPU-hour / episode stopping rule.

Per iteration: sync weights -> collect episodes (env + rollout workers) -> GRADE (grader labels each recorded
episode; agreement with the oracle is logged) -> relabel rewards (one terminal reward per episode) -> GAE ->
PPO update. Stops when (gpu_hours >= min_gpu_hours and episodes >= min_episodes) or gpu_hours >= max_gpu_hours.

Outputs in cfg.sip.run_dir:
    metrics.jsonl        one record per iteration (see _record)
    episodes.jsonl       one record per training episode: task, layout, oracle labels, grader label/score
    grading/iter_*/      grader manifests, outputs and logs
    episodes/iter_*/     recorded episodes (videos + meta), written by RecordingEnvWorker
    checkpoints          RLinf actor checkpoints (runner.logger.log_path/<experiment>/checkpoints/global_step_*)
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.utils.metric_utils import compute_evaluate_metrics

from sip_rl.grader_metrics import agreement
from sip_rl.graders import make_grader


def _gpu_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], timeout=10
        ).decode().split(",")
        return {"gpu_mem_used_mb": int(out[0]), "gpu_util": int(out[1])}
    except Exception:  # noqa: BLE001
        return {}


def _mean_or_none(xs):
    return float(np.mean(xs)) if len(xs) else None


def _reward_stats(reward: np.ndarray, oracle: np.ndarray) -> dict:
    """The reward the policy actually optimizes, split by the true outcome. With a perfect grader the mean on
    failures is 0 and on successes 1; optimism shows up as reward on failures."""
    fail, succ = reward[oracle == 0], reward[oracle == 1]
    return {
        "mean": float(reward.mean()),
        "mean_on_oracle_failure": _mean_or_none(fail),
        "mean_on_oracle_success": _mean_or_none(succ),
        "gap_success_minus_failure": (float(succ.mean() - fail.mean()) if len(fail) and len(succ) else None),
        "share_of_reward_on_failures": (float(fail.sum() / reward.sum()) if reward.sum() > 0 else None),
    }


class GradedRunner(EmbodiedRunner):
    def run(self):  # noqa: C901 (one linear loop, kept together for readability)
        sip = self.cfg.sip
        repo = pathlib.Path(sip.repo)
        run_dir = pathlib.Path(sip.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        grader_cfg = OmegaConf.to_container(sip.graders[sip.grader_name], resolve=True)
        grader = make_grader(grader_cfg, repo)
        # "label": reward = grader's 0/1 label; "score": reward = grader's success probability (oracle: same thing)
        reward_mode = grader_cfg.get("reward", "label")
        n_epochs = int(self.cfg.env.train.rollout_epoch)
        n_envs = int(self.cfg.env.train.total_num_envs)
        steps_per_ep = int(self.cfg.env.train.max_episode_steps)
        chunk = int(self.cfg.actor.model.num_action_chunks)
        min_h, max_h = float(sip.min_gpu_hours), float(sip.max_gpu_hours)
        min_eps = int(sip.min_episodes)
        snap_every_eps = int(sip.snapshot_every_episodes)
        snap_hours = sorted(float(h) for h in sip.get("snapshot_at_gpu_hours", []))
        n_gpus = int(sip.get("num_gpus", 1))

        # Resume: RLinf restored the model at self.global_step; drop log lines of later (lost) iterations and
        # continue the GPU-hour / episode counters from the last kept record.
        metrics_path = run_dir / "metrics.jsonl"
        for path in (metrics_path, run_dir / "episodes.jsonl"):
            if path.exists():
                kept = [line for line in path.read_text().splitlines() if json.loads(line)["iteration"] < self.global_step]
                path.write_text("".join(line + "\n" for line in kept))
        for it_dir in pathlib.Path(sip.episode_dir).glob("iter_*"):
            if int(it_dir.name.split("_")[1]) >= self.global_step:
                shutil.rmtree(it_dir, ignore_errors=True)
        prior = {"gpu_hours": 0.0, "episodes": 0, "env_steps": 0, "gradient_steps": 0}
        if metrics_path.exists() and metrics_path.read_text().strip():
            last = json.loads(metrics_path.read_text().strip().splitlines()[-1])
            prior = {k: last.get(k, 0) for k in prior}
        gpu_h, episodes, env_steps = prior["gpu_hours"], prior["episodes"], prior["env_steps"]
        grad_steps = prior["gradient_steps"]
        metrics_f = open(metrics_path, "a")  # noqa: SIM115
        episodes_f = open(run_dir / "episodes.jsonl", "a")  # noqa: SIM115
        (run_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(self.cfg, resolve=True))

        start_step, start_time = self.global_step, time.time()
        stop_reason = "max_steps"
        for _step in range(start_step, self.max_steps):
            t_iter = time.time()
            self.actor.set_global_step(self.global_step).wait()
            self.rollout.set_global_step(self.global_step).wait()
            self.env.set_global_step(self.global_step).wait()
            it = self.global_step
            times = {}

            t = time.time()
            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                times["sync_weights_s"] = time.time() - t

                t = time.time()
                with self.timer("generate_rollouts"):
                    env_handle = self.env.interact(
                        input_channel=self.env_channel, rollout_channel=self.rollout_channel, reward_channel=None
                    )
                    rollout_handle = self.rollout.generate(
                        input_channel=self.rollout_channel, output_channel=self.env_channel,
                        actor_channel=self.actor_channel,
                    )
                    self.actor.recv_rollout_trajectories(input_channel=self.actor_channel).wait()
                    rollout_handle.wait()
                times["rollout_s"] = time.time() - t

                # ---- grading phase: GPU is free here (actor and rollout models are offloaded) ----
                t = time.time()
                it_dir = pathlib.Path(sip.episode_dir) / f"iter_{it:05d}"
                eps = [json.loads(line) for line in (it_dir / "episodes.jsonl").read_text().splitlines()]
                results = grader.grade(eps, run_dir / "grading" / f"iter_{it:05d}")
                grader_wall = results.pop("__wall_s__", time.time() - t)
                grader_timing = results.pop("__timing__", {})
                # Batch column of an episode: rollout epochs are stacked epoch-major along the batch dimension.
                for e in eps:
                    e["batch_col"] = e["rollout_epoch"] * n_envs + e["env_index"]
                labels = [0.0] * (n_epochs * n_envs)
                for e in eps:
                    r = results[e["id"]]
                    e["reward"] = float(r["score"] if reward_mode == "score" else r["label"])
                    labels[e["batch_col"]] = e["reward"]
                # Mapping check: env reward per column must equal the recorded oracle outcome.
                layout = self.actor.episode_reward_sums().wait()[0]
                sums = layout["sums"]
                mismatches = sum(
                    int(round(sums[e["batch_col"]] / float(self.cfg.env.train.reward_coef))) != int(e["oracle_success_at_end"])
                    for e in eps
                )
                if mismatches or len(sums) != len(labels):
                    (run_dir / f"mapping_debug_iter{it}.json").write_text(json.dumps(
                        {"layout": layout, "oracle": {e["batch_col"]: int(e["oracle_success_at_end"]) for e in eps}}))
                    raise RuntimeError(f"episode->batch mapping check failed for {mismatches}/{len(eps)} episodes "
                                       f"(batch columns {len(sums)}, episodes {len(labels)})")
                self.actor.relabel_terminal_rewards(labels).wait()
                times["grading_s"] = time.time() - t
                times["grader_process_s"] = grader_wall

                t = time.time()
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = self.actor.compute_advantages_and_returns().wait()
                with self.timer("actor_training"):
                    actor_training_handle = self.actor.run_training()
                    actor_training_metrics = actor_training_handle.wait()
                times["update_s"] = time.time() - t
                self.global_step += 1

            # ---- bookkeeping ----
            # RLinf timers must be consumed every iteration; keep them as a finer time breakdown.
            timers = {f"runner/{k}": v for k, v in self.timer.consume_durations().items()}
            for prefix, handle in (("env", env_handle), ("rollout", rollout_handle), ("actor", actor_training_handle)):
                try:
                    timers.update({f"{prefix}/{k}": v for k, v in handle.consume_durations().items()})
                except Exception as exc:  # noqa: BLE001
                    timers[f"{prefix}/error"] = repr(exc)
            iter_s = time.time() - t_iter
            gpu_h += iter_s * n_gpus / 3600
            episodes += len(eps)
            env_steps += len(eps) * steps_per_ep
            oracle = np.array([int(e["oracle_success_at_end"]) for e in eps])
            lab = np.array([int(results[e["id"]]["label"]) for e in eps])
            scores = [results[e["id"]].get("score") for e in eps]
            score = np.array(scores, dtype=float) if all(s is not None for s in scores) else None
            tasks = [e["task_id"] for e in eps]
            for e in eps:
                r = results[e["id"]]
                episodes_f.write(json.dumps({
                    "iteration": it, "id": e["id"], "task_id": e["task_id"], "task": e["task"], "trial_id": e["trial_id"],
                    "oracle_success_at_end": e["oracle_success_at_end"], "oracle_success_once": e["oracle_success_once"],
                    "first_success_chunk": e["first_success_chunk"], "grader_label": int(r["label"]),
                    "grader_score": r.get("score"), "reward": e["reward"], "final_state": e.get("final_state"),
                    "agentview": e["agentview"], "wrist": e.get("wrist"),
                    "grader_extra": {k: v for k, v in r.items() if k not in ("label", "score")},
                }) + "\n")
            episodes_f.flush()

            env_results = [x for x in env_handle.wait() if x is not None]
            env_metrics = compute_evaluate_metrics(env_results) if env_results else {}
            per_task_success = {}
            for task_id in sorted(set(tasks)):
                m = np.array(tasks) == task_id
                per_task_success[str(task_id)] = float(oracle[m].mean())
            record = {
                "iteration": it,
                "episodes": episodes,
                "env_steps": env_steps,
                "gpu_hours": gpu_h,
                "wall_clock_s": time.time() - start_time,
                "iteration_s": iter_s,
                **times,
                "timers": timers,
                "train/oracle_success_rate": float(oracle.mean()),
                "train/oracle_success_once_rate": float(np.mean([bool(e["oracle_success_once"]) for e in eps])),
                "train/per_task_oracle_success": per_task_success,
                "grader": grader.name,
                "grader_timing": grader_timing,
                "grader_vs_oracle": agreement(oracle, lab, score, tasks),
                "reward_mode": reward_mode,
                "reward": _reward_stats(np.array([e["reward"] for e in eps]), oracle),
                "train/steps_to_success_once": _mean_or_none(
                    [e["first_success_chunk"] * chunk for e in eps if e["first_success_chunk"] is not None]),
                "env": {k: float(v) for k, v in env_metrics.items() if np.isscalar(v) or getattr(v, "ndim", 1) == 0},
                "rollout": self._aggregate_numeric_metrics(actor_rollout_metrics),
                "train": self._aggregate_numeric_metrics(actor_training_metrics),
                **_gpu_snapshot(),
            }
            # Optimizer steps this iteration: update_epoch passes over (chunk steps x episodes) samples.
            samples = (steps_per_ep // int(self.cfg.actor.model.num_action_chunks)) * len(eps)
            grad_steps_iter = int(self.cfg.algorithm.update_epoch) * (samples // int(self.cfg.actor.global_batch_size))
            grad_steps += grad_steps_iter
            record["gradient_steps"] = grad_steps
            record["gradient_steps_iter"] = grad_steps_iter
            record["train_samples"] = samples

            # ---- checkpoint + stopping rule ----
            done_min = gpu_h >= min_h and episodes >= min_eps
            if done_min:
                stop_reason = "min_gpu_hours_and_min_episodes"
            elif gpu_h >= max_h:
                stop_reason = "max_gpu_hours"
            stop = done_min or gpu_h >= max_h or self.global_step >= self.max_steps
            prev_eps, prev_h = episodes - len(eps), gpu_h - iter_s * n_gpus / 3600
            crossed_eps = episodes // snap_every_eps > prev_eps // snap_every_eps
            crossed_h = [h for h in snap_hours if prev_h < h <= gpu_h]
            if stop or crossed_eps or crossed_h:
                # Small snapshot for evaluation (kept), full RLinf checkpoint for resuming (only the latest kept).
                snap = run_dir / "snapshots" / f"step_{self.global_step:05d}.pt"
                self.actor.save_trainable(str(snap)).wait()
                self._save_checkpoint()
                ckpt_root = pathlib.Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name / "checkpoints"
                for old in ckpt_root.glob("global_step_*"):
                    if old.name != f"global_step_{self.global_step}":
                        shutil.rmtree(old, ignore_errors=True)
                record["snapshot"] = str(snap)
                record["snapshot_reasons"] = (["episodes"] if crossed_eps else []) + [f"gpu_h>={h}" for h in crossed_h] + (["end"] if stop else [])
            metrics_f.write(json.dumps(record, default=float) + "\n")
            metrics_f.flush()
            flat = {k: v for k, v in record.items() if isinstance(v, (int, float)) and v is not None}
            flat.update({f"grader/{k}": v for k, v in record["grader_vs_oracle"].items() if isinstance(v, (int, float))})
            self.metric_logger.log(flat, it)
            self.logger.info(
                f"[iter {it}] episodes={episodes} gpu_h={gpu_h:.2f} oracle_success={oracle.mean():.3f} "
                f"grader_success={lab.mean():.3f} acc={record['grader_vs_oracle']['accuracy']:.3f} "
                f"reward_fail={record['reward']['mean_on_oracle_failure']} reward_succ={record['reward']['mean_on_oracle_success']} "
                f"times={ {k: round(v, 1) for k, v in times.items()} }"
            )
            if stop:
                break

        (run_dir / "run_end.json").write_text(json.dumps({
            "stop_reason": stop_reason, "gpu_hours": gpu_h, "episodes": episodes, "env_steps": env_steps,
            "global_step": self.global_step, "wall_clock_s": time.time() - start_time,
        }))
        metrics_f.close()
        episodes_f.close()
        grader.close()
        self._finish_run()
