"""One round of full fine-tuning: a checkpoint plus a LeRobot dataset in, a new checkpoint out (openpi JAX).

Runs in the openpi environment. Builds an openpi TrainConfig in code (openpi stays unmodified) and runs openpi's own
scripts/train.py main(). The config is pi05_libero's with the round changes: the dataset is the round's kept
episodes, batch 32, a short cosine schedule with warmup at most 10% of the steps, no EMA (a 0.999 average over a
few hundred steps would serve weights that barely moved), and the start weights are the previous round's params.
Norm stats are the official LIBERO ones shipped with pi05_libero; they are copied into the checkpoint's assets so
openpi's stock serve_policy.py can serve the result with --policy.config pi05_libero.

    third_party/openpi/.venv/bin/python scripts/finetune_round.py --config configs/rft/middle_bowl.json \
        --dataset-home runs/x/round_1/lerobot --repo-id rft/train \
        --init-checkpoint checkpoints/pi05_libero --out runs/x/round_1/checkpoint --exp-name round_1

The final checkpoint directory is printed last and written to <out>/train_info.json ("final_checkpoint").
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rft.config import LoopConfig  # noqa: E402

CONFIG_NAME = "pi05_rft"
LIBERO_ASSET_ID = "physical-intelligence/libero"


def build_train_config(cfg: LoopConfig, repo_id: str, init_checkpoint: pathlib.Path, out: pathlib.Path, exp_name: str):
    from openpi.models import pi0_config  # noqa: PLC0415
    from openpi.training import config as _config  # noqa: PLC0415
    from openpi.training import optimizer as _optimizer  # noqa: PLC0415
    from openpi.training import weight_loaders  # noqa: PLC0415

    t = cfg.train
    params = init_checkpoint / "params"
    if not params.exists():
        raise FileNotFoundError(f"no params/ under {init_checkpoint}")
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=_config.LeRobotLiberoDataConfig(
            repo_id=repo_id,
            base_config=_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            assets=_config.AssetsConfig(assets_dir=str(REPO / cfg.norm_stats_dir), asset_id=LIBERO_ASSET_ID),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=t.warmup_steps, peak_lr=t.peak_lr, decay_steps=t.steps, decay_lr=t.decay_lr
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        batch_size=t.batch_size,
        num_train_steps=t.steps,
        num_workers=t.num_workers,
        log_interval=10,
        save_interval=t.save_interval,
        keep_period=None,
        checkpoint_base_dir=str(out),
        seed=t.seed,
        wandb_enabled=False,
        overwrite=True,
    )


def load_openpi_train():
    path = REPO / "third_party" / "openpi" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_checkpoint(train_config) -> pathlib.Path:
    ckpt_dir = pathlib.Path(train_config.checkpoint_dir)
    steps = sorted(int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"no step directories under {ckpt_dir}")
    return ckpt_dir / str(steps[-1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=pathlib.Path)
    p.add_argument("--dataset-home", required=True, type=pathlib.Path, help="parent of <repo_id>, becomes HF_LEROBOT_HOME")
    p.add_argument("--repo-id", default="rft/train")
    p.add_argument("--init-checkpoint", required=True, type=pathlib.Path, help="directory containing params/")
    p.add_argument("--out", required=True, type=pathlib.Path, help="checkpoint base dir for this round")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--steps", type=int, default=None, help="override the config's training steps (smoke tests)")
    args = p.parse_args()

    cfg = LoopConfig.load(args.config)
    if args.steps is not None:
        cfg.train.steps = args.steps
        cfg.train.warmup_steps = min(cfg.train.warmup_steps, max(1, args.steps // 10))
        cfg.train.save_interval = min(cfg.train.save_interval, args.steps)
    dataset_home = args.dataset_home.resolve()
    if not (dataset_home / args.repo_id / "meta" / "info.json").exists():
        raise FileNotFoundError(f"no LeRobot dataset at {dataset_home / args.repo_id}")
    # lerobot reads HF_LEROBOT_HOME at import time, so it must be set before openpi is imported.
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home)

    train_config = build_train_config(cfg, args.repo_id, args.init_checkpoint.resolve(), args.out.resolve(), args.exp_name)
    openpi_train = load_openpi_train()
    t0 = time.time()
    openpi_train.main(train_config)
    final = final_checkpoint(train_config)
    info = {
        "final_checkpoint": str(final),
        "init_checkpoint": str(args.init_checkpoint.resolve()),
        "dataset": str(dataset_home / args.repo_id),
        "train": cfg.train.__dict__,
        "train_s": time.time() - t0,
        "config_name": CONFIG_NAME,
        "exp_name": args.exp_name,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train_info.json").write_text(json.dumps(info, indent=2))
    print(str(final))


if __name__ == "__main__":
    main()
