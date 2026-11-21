"""Experiment 1 training config: LoRA fine-tuning of pi05_libero on one of our LeRobot datasets.

Importing this module registers the config "pi05_libero_lora" in openpi's config table, so both training
(train.py here) and serving (serve.py) can refer to it by name. Every Experiment 1 row uses this config; only the
dataset (--data.repo-id) and the run name (--exp-name) change.

Choices, and why:
  start from pi05_libero (not pi05_base)  the policy that produced the pool; we measure what fine-tuning on its own
                                          (graded) episodes does to it
  LoRA on both experts                    openpi's supported low-memory path; same filters as its pi0 LoRA example
  norm stats from pi05_libero             the fine-tuned policy keeps the base model's normalization
  no EMA                                  openpi turns EMA off for LoRA
"""
import os

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

BASE = "gs://openpi-assets/checkpoints/pi05_libero"
_model = dict(pi05=True, action_horizon=10, discrete_state_input=False,
              paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")

CONFIG = _config.TrainConfig(
    name="pi05_libero_lora",
    model=pi0_config.Pi0Config(**_model),
    data=_config.LeRobotLiberoDataConfig(
        repo_id="exp1/clean",  # override with --data.repo-id
        base_config=_config.DataConfig(prompt_from_task=True),
        extra_delta_transform=False,  # same as pi05_libero: LIBERO actions are already deltas
        assets=_config.AssetsConfig(assets_dir=BASE + "/assets", asset_id="physical-intelligence/libero"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(BASE + "/params"),
    freeze_filter=pi0_config.Pi0Config(**_model).get_freeze_filter(),
    ema_decay=None,
    batch_size=32,
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-5, decay_steps=2_000, decay_lr=5e-6),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    num_train_steps=2_000,
    log_interval=50,
    save_interval=500,
    keep_period=None,
    wandb_enabled=False,
    checkpoint_base_dir=os.path.join(os.environ.get("EXP", "."), "checkpoints"),
    assets_base_dir=os.path.join(os.environ.get("EXP", "."), "assets"),
)

if CONFIG.name not in _config._CONFIGS_DICT:
    _config._CONFIGS.append(CONFIG)
    _config._CONFIGS_DICT[CONFIG.name] = CONFIG
