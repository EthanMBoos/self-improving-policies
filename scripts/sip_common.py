"""Shared openpi configs and data helpers for this project (run with openpi's environment).

Configs live here instead of inside third_party/openpi so openpi stays unmodified.
"""

import collections
import dataclasses
import json
import pathlib

import numpy as np
import torch.utils.data

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import weight_loaders
import openpi.transforms as _transforms

REPO = pathlib.Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints"
LIBERO_REPO_ID = "physical-intelligence/libero"

# The official LIBERO norm stats bundled with the pi05_libero checkpoint (computed on the same dataset). Using them
# avoids the norm-stats mismatch behind many failed pi0.5 LoRA reproductions (openpi issue #711).
LIBERO_ASSETS = _config.AssetsConfig(assets_dir=str(CKPT / "pi05_libero" / "assets"), asset_id=LIBERO_REPO_ID)

LORA_MODEL = pi0_config.Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",  # LoRA rank 16
    action_expert_variant="gemma_300m_lora",  # LoRA rank 32
)


def _libero_data() -> _config.LeRobotLiberoDataConfig:
    return _config.LeRobotLiberoDataConfig(
        repo_id=LIBERO_REPO_ID,
        base_config=_config.DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        assets=LIBERO_ASSETS,
    )


CONFIGS = {
    # LoRA SFT on the LIBERO demos. Matches github.com/imwaterhuang/pi05-libero-lora (88% LIBERO-Spatial at 5k x 32):
    # openpi's default LR schedule (1k warmup, 2.5e-5 peak, cosine over 30k), AdamW, no EMA. openpi's LoRA freeze
    # filter freezes the Gemma / action-expert weights; LoRA adapters, the SigLIP vision encoder and the action/time
    # projections are trained.
    "pi05_libero_lora": _config.TrainConfig(
        name="pi05_libero_lora",
        model=LORA_MODEL,
        data=_libero_data(),
        freeze_filter=LORA_MODEL.get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(CKPT / "pi05_base" / "params")),
        num_train_steps=5_000,
        save_interval=500,
        keep_period=500,
        log_interval=10,
        num_workers=8,
        checkpoint_base_dir=str(CKPT),
        wandb_enabled=False,
    ),
    # openpi's official full fine-tune config (for serving the released pi05_libero checkpoint).
    "pi05_libero": _config.get_config("pi05_libero"),
    # pi05_base zero-shot with the LIBERO pipeline. Keeps pi0.5's default discrete_state_input=True (how the base
    # model was pretrained); pi05_base ships no LIBERO norm stats, so serving uses LIBERO_ASSETS.
    "pi05_base_zeroshot": dataclasses.replace(
        _config.get_config("pi05_libero"),
        name="pi05_base_zeroshot",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
        data=_libero_data(),
    ),
}


def get_config(name: str) -> _config.TrainConfig:
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    return CONFIGS[name]


# ---------------------------------------------------------------------------------------------------------------------
# Train / validation split by episode
# ---------------------------------------------------------------------------------------------------------------------


def split_episodes(repo_id: str, val_per_task: int, seed: int = 0) -> dict:
    """Hold out `val_per_task` demos of every task (chosen with a fixed seed) for validation."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    by_task = collections.defaultdict(list)
    for ep in meta.episodes.values():
        by_task[ep["tasks"][0]].append(ep["episode_index"])
    rng = np.random.default_rng(seed)
    val = sorted(int(e) for eps in by_task.values() for e in rng.choice(sorted(eps), val_per_task, replace=False))
    train = sorted(set(meta.episodes) - set(val))
    lengths = {ep["episode_index"]: ep["length"] for ep in meta.episodes.values()}
    return {
        "repo_id": repo_id,
        "seed": seed,
        "val_per_task": val_per_task,
        "num_tasks": len(by_task),
        "train_episodes": train,
        "val_episodes": val,
        "train_frames": int(sum(lengths[e] for e in train)),
        "val_frames": int(sum(lengths[e] for e in val)),
    }


class EpisodeSubset(torch.utils.data.Dataset):
    """Restricts a full LeRobotDataset to the frames of some episodes.

    We don't use LeRobotDataset(episodes=...): in the pinned lerobot version it indexes episode boundaries by global
    episode index into a subset-sized table, which corrupts action chunks.
    """

    def __init__(self, dataset, episodes: list[int]):
        self._dataset = dataset
        idx = dataset.episode_data_index
        self._frames = np.concatenate(
            [np.arange(int(idx["from"][e]), int(idx["to"][e])) for e in episodes]
        )

    def __getitem__(self, i):
        return self._dataset[int(self._frames[i])]

    def __len__(self):
        return len(self._frames)


def create_episode_dataset(data_config: _config.DataConfig, action_horizon: int, episodes: list[int] | None):
    """Like openpi's data_loader.create_torch_dataset, restricted to `episodes` (None = all)."""
    from lerobot.common.datasets import lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys},
    )
    # Full dataset: episode_data_index row e must be global episode e.
    assert [int(x) for x in dataset.hf_dataset.select([int(dataset.episode_data_index["from"][e]) for e in (0, len(meta.episodes) - 1)])["episode_index"]] == [0, len(meta.episodes) - 1]
    if episodes is not None:
        dataset = EpisodeSubset(dataset, episodes)
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)])
    return dataset


def write_json(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
