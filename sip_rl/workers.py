"""RLinf worker subclasses for grader-in-the-loop RL.

RecordingEnvWorker: writes every training episode to disk (upright third-person and wrist frames at action-chunk
boundaries as MP4, plus metadata including the simulator's oracle success flags) so a grader can label it after
the rollout phase. Requires fixed-length episodes (env.train.ignore_terminations=True, auto_reset=False).

GradedActor: replaces the env rewards in the collected batch by one terminal reward per episode (the grader's
label) before advantages are computed. All runs, oracle included, use this same reward shape, so the reward
source is the only difference between runs.
"""

from __future__ import annotations

import gc
import json
import pathlib
import time

import imageio
import numpy as np
import torch

from rlinf.envs.utils import get_env_attr
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker


def _np_frames(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(np.uint8)


class RecordingEnvWorker(EnvWorker):
    def _rec_setup(self):
        if getattr(self, "_rec_ready", False):
            return
        self._rec_root = pathlib.Path(self.cfg.sip.episode_dir)
        self._rec_fps = float(self.cfg.sip.get("video_fps", 2.0))
        self._rec = {}
        self._rec_step = None
        self._rec_epoch = 0
        self._rec_ready = True

    def bootstrap_step(self):
        outputs = super().bootstrap_step()
        self._rec_setup()
        if self._rec_step != self._trajectory_step:
            self._rec_step, self._rec_epoch = self._trajectory_step, 0
        for stage_id, out in enumerate(outputs):
            env = self.env_list[stage_id]
            obs = out.obs
            self._rec[stage_id] = {
                "agent": [_np_frames(obs["main_images"])],
                "wrist": [_np_frames(obs["wrist_images"])] if obs.get("wrist_images") is not None else None,
                "success_once": [],
                "task_ids": np.asarray(get_env_attr(env, "task_ids")).copy(),
                "trial_ids": np.asarray(get_env_attr(env, "trial_ids")).copy(),
                "tasks": list(obs["task_descriptions"]),
                "t0": time.time(),
            }
        return outputs

    def env_interact_step(self, chunk_actions, stage_id):
        env_output, env_info, payload = super().env_interact_step(chunk_actions, stage_id)
        rec = self._rec.get(stage_id) if getattr(self, "_rec_ready", False) else None
        if rec is None:
            return env_output, env_info, payload
        obs = env_output.obs
        rec["agent"].append(_np_frames(obs["main_images"]))
        if rec["wrist"] is not None:
            rec["wrist"].append(_np_frames(obs["wrist_images"]))
        episode = (env_output.env_infos or {}).get("episode", {})
        if "success_once" in episode:
            rec["success_once"].append(np.asarray(episode["success_once"].cpu()).astype(bool))
        trunc = env_output.transition.truncations
        if trunc is not None and bool(trunc[:, -1].all()):
            self._flush(stage_id, episode)
        return env_output, env_info, payload

    def _final_sim_states(self, stage_id: int):
        """Flattened MuJoCo state of every env at the end of the episode (same format as LIBERO init states), so
        which goal predicates hold and how far objects are from their targets can be computed offline."""
        obj = self.env_list[stage_id]
        for _ in range(8):
            inner = getattr(obj, "env", None)
            if inner is None:
                return None
            if type(inner).__name__ == "ReconfigureSubprocEnv":
                try:
                    return [np.asarray(s, dtype=np.float64) for s in inner.get_sim_state()]
                except Exception:  # noqa: BLE001
                    return None
            obj = inner
        return None

    def _flush(self, stage_id: int, episode: dict):
        rec = self._rec.pop(stage_id)
        final_states = self._final_sim_states(stage_id)
        agent = np.stack(rec["agent"], axis=1)  # [N, T, H, W, 3]
        wrist = np.stack(rec["wrist"], axis=1) if rec["wrist"] is not None else None
        succ = np.stack(rec["success_once"], axis=1) if rec["success_once"] else None  # [N, chunks]
        at_end = np.asarray(episode["success_at_end"].cpu()).astype(bool) if "success_at_end" in episode else None
        n = agent.shape[0]
        it_dir = self._rec_root / f"iter_{self._rec_step:05d}"
        it_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(n):
            env_index = (self._rank * self.stage_num + stage_id) * n + i
            eid = f"it{self._rec_step:05d}_ep{self._rec_epoch}_env{env_index:03d}"
            d = it_dir / eid
            d.mkdir(exist_ok=True)
            imageio.mimwrite(d / "agentview.mp4", agent[i], fps=self._rec_fps, macro_block_size=1)
            if wrist is not None:
                imageio.mimwrite(d / "wrist.mp4", wrist[i], fps=self._rec_fps, macro_block_size=1)
            once = bool(succ[i].any()) if succ is not None else None
            first = int(np.argmax(succ[i])) if once else None
            if final_states is not None:
                np.save(d / "final_state.npy", final_states[i])
            meta = {
                "id": eid,
                "iteration": self._rec_step,
                "rollout_epoch": self._rec_epoch,
                "env_index": env_index,
                "task_id": int(rec["task_ids"][i]),
                "trial_id": int(rec["trial_ids"][i]),
                "task": rec["tasks"][i],
                "num_frames": int(agent.shape[1]),
                "oracle_success_at_end": bool(at_end[i]) if at_end is not None else once,
                "oracle_success_once": once,
                "first_success_chunk": first,
                "agentview": str(d / "agentview.mp4"),
                "wrist": str(d / "wrist.mp4") if wrist is not None else None,
                "final_state": str(d / "final_state.npy") if final_states is not None else None,
                "episode_wall_s": time.time() - rec["t0"],
            }
            (d / "meta.json").write_text(json.dumps(meta))
            lines.append(json.dumps(meta))
        with open(it_dir / "episodes.jsonl", "a") as f:
            f.write("\n".join(lines) + "\n")
        self._rec_epoch += 1


class GradedActor(EmbodiedFSDPActor):
    # One GPU holds the actor, the rollout model and the env renderers in turn, so memory the actor no longer
    # needs is handed back explicitly: the trajectory batch after training, cached blocks after weight sync.
    def _log_mem(self, where: str) -> None:
        self.log_info(f"[sip mem] actor {where}: allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB "
                      f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB")

    def _clear_fsdp_saved_views(self) -> int:
        """FSDP (use_orig_params=True) saves per-parameter views in flat_param._tensors during forward and clears
        them in the post-backward hook. Frozen units (VLM, frozen expert weights) never run backward, so their
        views would keep the GPU copy of the weights alive after offload (~6.5 GB). They are only needed between
        a forward and its backward, so clearing them after the training step is safe."""
        cleared = 0
        for module in self.model.modules():
            handle = getattr(module, "_handle", None)
            tensors = getattr(getattr(handle, "flat_param", None), "_tensors", None)
            if tensors:
                for i in range(len(tensors)):
                    if tensors[i] is not None:
                        tensors[i] = None
                        cleared += 1
        return cleared

    def run_training(self):
        metrics = super().run_training()
        self.rollout_batch = {}
        self._clear_fsdp_saved_views()
        gc.collect()
        torch.cuda.empty_cache()
        self._log_mem("after training")
        return metrics

    async def sync_model_to_rollout(self) -> None:
        await super().sync_model_to_rollout()
        gc.collect()
        torch.cuda.empty_cache()
        self._log_mem("after sync+offload")

    def save_trainable(self, path: str) -> dict:
        """Save only the trainable tensors (LoRA, value head, small heads): a few tens of MB per snapshot."""
        state = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                clean = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
                state[clean] = p.detach().to("cpu", copy=True)
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        return {"tensors": len(state), "params": int(sum(v.numel() for v in state.values()))}

    # Batch layout (fixed-length episodes): rewards are [T chunk steps, B, actions per chunk] with
    # B = rollout_epoch * num_envs, epoch-major; each column is one complete episode.

    def episode_reward_sums(self) -> dict:
        """Env reward summed per batch column; with relative rewards this equals success at the final step."""
        r = self.rollout_batch["rewards"]
        return {"shape": list(r.shape), "sums": r.float().sum(dim=(0, 2)).cpu().tolist()}

    def relabel_terminal_rewards(self, labels: list[float]) -> dict:
        """rewards := 0 everywhere except the last action of each episode (column), which gets labels[column]."""
        r = self.rollout_batch["rewards"]
        lab = torch.as_tensor(labels, dtype=r.dtype, device=r.device)
        if lab.shape != (r.shape[1],):
            raise ValueError(f"expected {r.shape[1]} labels (one per batch column), got {tuple(lab.shape)}")
        new = torch.zeros_like(r)
        new[-1, :, -1] = lab
        self.rollout_batch["rewards"] = new
        return {"episodes": int(lab.numel()), "positive": float(lab.sum())}
