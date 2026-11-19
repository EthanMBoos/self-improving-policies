"""Loop configuration. Every pre-registered number lives in a JSON file loaded here; scripts never hardcode them.

Example: configs/rft/middle_bowl.json. Load with LoopConfig.load(path); nested sections are plain dataclasses.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any


@dataclasses.dataclass
class TrainSettings:
    """One round of full fine-tuning (openpi JAX). Short-round defaults: no EMA, warmup at most 10% of steps."""

    steps: int = 1000
    batch_size: int = 32
    peak_lr: float = 5e-5
    warmup_steps: int = 100
    decay_lr: float = 5e-6
    num_workers: int = 8
    save_interval: int = 500
    seed: int = 42


@dataclasses.dataclass
class GraderSettings:
    """The VLM grader. `threshold` is the pass mark on its last-frame success probability (non-negotiable 6)."""

    name: str = "robometer"
    model_dir: str = "checkpoints/Robometer-4B-infer"
    python: str = "third_party/robometer/.venv/bin/python"
    threshold: float = 0.65
    audit_thresholds: list[float] = dataclasses.field(default_factory=lambda: [0.5, 0.65, 0.85])
    max_frames: int = 8
    camera: str = "agentview_grader"  # Robometer's LIBERO training orientation; see scripts/make_grader_videos.py
    batch_size: int = 16


@dataclasses.dataclass
class LoopConfig:
    run_name: str
    suite: str = "libero_90"
    task_id: int = 0
    task_name: str = ""  # informational; the worker reads the instruction from LIBERO
    rounds: int = 3
    rollouts_per_round: int = 300
    min_kept: int = 50
    exam_episodes: int = 100
    exam_seed_base: int = 100000  # exam seeds never overlap rollout seeds
    rollout_seed_base: int = 1000
    rollout_seed_stride: int = 1000  # round k uses rollout_seed_base + k * stride + i
    continue_from_previous: bool = True  # round k+1 starts from round k's checkpoint (else from the base)
    log_full_dataset: bool = True  # also pack all rollouts (not just the kept ones) into a LeRobot dataset
    base_checkpoint: str = "checkpoints/pi05_libero"
    norm_stats_dir: str = "checkpoints/pi05_libero/assets"
    policy_config: str = "pi05_libero"  # openpi config used to serve every checkpoint of this loop
    port: int = 8001
    replan_steps: int = 5
    wait_steps: int = 10
    resize: int = 224
    train: TrainSettings = dataclasses.field(default_factory=TrainSettings)
    grader: GraderSettings = dataclasses.field(default_factory=GraderSettings)

    def rollout_seeds(self, round_index: int) -> list[int]:
        base = self.rollout_seed_base + round_index * self.rollout_seed_stride
        return list(range(base, base + self.rollouts_per_round))

    def exam_seeds(self) -> list[int]:
        return list(range(self.exam_seed_base, self.exam_seed_base + self.exam_episodes))

    def validate(self) -> None:
        if self.rollouts_per_round > self.rollout_seed_stride:
            raise ValueError("rollouts_per_round must not exceed rollout_seed_stride, or rounds would share seeds")
        last_rollout_seed = self.rollout_seed_base + self.rounds * self.rollout_seed_stride
        if last_rollout_seed >= self.exam_seed_base:
            raise ValueError("exam seeds overlap rollout seeds")
        if not 0.0 <= self.grader.threshold <= 1.0:
            raise ValueError("grader threshold must be a probability")
        if self.train.warmup_steps > self.train.steps // 10:
            raise ValueError("warmup must be at most 10% of the steps (short-round rule)")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoopConfig":
        d = dict(d)
        train = TrainSettings(**d.pop("train", {}))
        grader = GraderSettings(**d.pop("grader", {}))
        cfg = cls(train=train, grader=grader, **d)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "LoopConfig":
        return cls.from_dict(json.loads(pathlib.Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
