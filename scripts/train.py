"""Train a pi0.5 policy with openpi (LoRA SFT by default), logging metrics for comparing training methods.

Run from the repo root with openpi's environment:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 third_party/openpi/.venv/bin/python scripts/train.py \
        --config pi05_libero_lora --exp-name sft_5k

Outputs in checkpoints/<config>/<exp-name>/:
    <step>/          openpi checkpoints (params + norm stats), servable with scripts/serve_policy.py
    run_info.json    config, code versions, hardware, data split sizes, parameter counts
    data_split.json  train / held-out validation episode indices
    metrics.jsonl    {"kind": "train"} every --log-interval steps, {"kind": "val"} every --val-interval steps

Train metrics: loss, grad_norm, param_norm, lr, samples_seen, epochs, step_time_s, data_wait_s, samples_per_s,
wall_clock_s, gpu_hours, gpu_mem_peak_gib. Val metrics, on fixed frames from held-out demos ("val/") and from
training demos ("train_subset/", same size, to measure the generalization gap): flow-matching loss, and the error of
actions sampled the way they are at inference (10 flow steps) against the demo actions: MSE / L1 over the action
chunk, L1 of the first (executed) action, and gripper open/close agreement.
"""

import argparse
import dataclasses
import functools
import json
import logging
import platform
import subprocess
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

import sip_common

# openpi's own training script, imported as a module for init_train_state / train_step.
_spec = __import__("importlib.util").util.spec_from_file_location(
    "openpi_train", sip_common.REPO / "third_party/openpi/scripts/train.py"
)
openpi_train = __import__("importlib.util").util.module_from_spec(_spec)
_spec.loader.exec_module(openpi_train)

ACTION_DIM = 7  # LIBERO: 6 end-effector deltas + gripper; model actions are zero-padded to 32
GRIPPER = 6


def _git_rev(path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _num_params(tree) -> int:
    return int(sum(x.size for x in jax.tree.leaves(tree)))


def _fixed_batches(dataset, n_frames: int, batch_size: int, sharding_, seed: int):
    """Loads n_frames fixed, evenly spread frames once and returns them as device batches."""
    idx = np.linspace(0, len(dataset) - 1, n_frames).astype(int)
    np.random.default_rng(seed).shuffle(idx)  # mix episodes within each batch
    batches = []
    for i in range(0, n_frames - batch_size + 1, batch_size):
        items = [dataset[int(j)] for j in idx[i : i + batch_size]]
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *items)
        obs, actions = _model.Observation.from_dict(batch), batch["actions"]
        batches.append(jax.device_put((obs, actions), sharding_))
    return batches


def _eval_step(state, batch, rng):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    obs, actions = batch
    loss_rng, sample_rng = jax.random.split(rng)
    flow_loss = jnp.mean(model.compute_loss(loss_rng, obs, actions, train=False))
    pred = model.sample_actions(sample_rng, obs, num_steps=10)
    a, p = actions[..., :ACTION_DIM], pred[..., :ACTION_DIM]
    return {
        "flow_loss": flow_loss,
        "action_mse": jnp.mean((p - a) ** 2),
        "action_l1": jnp.mean(jnp.abs(p - a)),
        "first_action_l1": jnp.mean(jnp.abs(p[:, 0] - a[:, 0])),
        "gripper_agreement": jnp.mean(jnp.sign(p[..., GRIPPER]) == jnp.sign(a[..., GRIPPER])),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pi05_libero_lora", choices=sorted(sip_common.CONFIGS))
    p.add_argument("--exp-name", required=True)
    p.add_argument("--num-train-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--save-interval", type=int)
    p.add_argument("--val-per-task", type=int, default=2, help="held-out demos per task (0 = no validation)")
    p.add_argument("--val-frames", type=int, default=512, help="fixed frames per validation pass (each split)")
    p.add_argument("--val-interval", type=int, default=250)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    overrides = {"exp_name": args.exp_name, "overwrite": args.overwrite, "resume": args.resume}
    for field in ("num_train_steps", "batch_size", "save_interval"):
        if getattr(args, field) is not None:
            overrides[field] = getattr(args, field)
    config = dataclasses.replace(sip_common.get_config(args.config), **overrides)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    jax.config.update("jax_compilation_cache_dir", str(sip_common.pathlib.Path("~/.cache/jax").expanduser()))

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))

    ckpt_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume
    )
    run_dir = config.checkpoint_dir

    # --- data: hold out demos per task, train on the rest -------------------------------------------------------
    data_config = config.data.create(config.assets_dirs, config.model)
    split_path = run_dir / "data_split.json"
    if resuming and split_path.exists():
        split = json.loads(split_path.read_text())
    elif args.val_per_task > 0:
        split = sip_common.split_episodes(data_config.repo_id, args.val_per_task)
    else:
        split = sip_common.split_episodes(data_config.repo_id, 0)
    sip_common.write_json(split_path, split)
    logging.info(
        f"train: {len(split['train_episodes'])} demos / {split['train_frames']} frames, "
        f"val: {len(split['val_episodes'])} demos / {split['val_frames']} frames"
    )

    # openpi's loader builds its dataset via create_torch_dataset; restrict it to the training episodes.
    _data_loader.create_torch_dataset = lambda dc, horizon, model_config: sip_common.create_episode_dataset(
        dc, horizon, split["train_episodes"]
    )
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)

    val_sets = {}
    if split["val_episodes"]:
        for name, episodes in (("val", split["val_episodes"]), ("train_subset", split["train_episodes"])):
            ds = _data_loader.transform_dataset(
                sip_common.create_episode_dataset(data_config, config.model.action_horizon, episodes), data_config
            )
            val_sets[name] = _fixed_batches(ds, args.val_frames, config.batch_size, data_sharding, seed=0)
        logging.info(f"validation: {args.val_frames} fixed frames per split")

    # --- model -------------------------------------------------------------------------------------------------
    train_state, state_sharding = openpi_train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(ckpt_manager, train_state, data_loader)
    n_total = _num_params(train_state.params)
    n_trainable = _num_params(train_state.params.filter(config.trainable_filter))
    logging.info(f"params: {n_total / 1e9:.3f} B total, {n_trainable / 1e6:.1f} M trainable")

    sip_common.write_json(
        run_dir / "run_info.json",
        {
            "config_name": args.config,
            "exp_name": args.exp_name,
            "config": {f.name: repr(getattr(config, f.name)) for f in dataclasses.fields(config)},
            "args": vars(args),
            "repo_commit": _git_rev(sip_common.REPO),
            "openpi_commit": _git_rev(sip_common.REPO / "third_party/openpi"),
            "host": platform.node(),
            "devices": [d.device_kind for d in jax.devices()],
            "jax_version": jax.__version__,
            "params_total": n_total,
            "params_trainable": n_trainable,
            "norm_stats": str(sip_common.LIBERO_ASSETS.assets_dir),
            "train_episodes": len(split["train_episodes"]),
            "train_frames": split["train_frames"],
            "val_episodes": len(split["val_episodes"]),
            "val_frames": split["val_frames"],
        },
    )

    ptrain_step = jax.jit(
        functools.partial(openpi_train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(_eval_step)
    lr_schedule = config.lr_schedule.create()
    n_gpus = jax.device_count()

    metrics_file = open(run_dir / "metrics.jsonl", "a")  # noqa: SIM115

    def log(record):
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    def peak_mem_gib():
        stats = jax.devices()[0].memory_stats() or {}
        return stats.get("peak_bytes_in_use", 0) / 2**30

    def validate(step, wall):
        if not val_sets:
            return
        t0 = time.time()
        record = {"kind": "val", "step": step, "samples_seen": step * config.batch_size, "wall_clock_s": wall}
        rng = jax.random.key(1234)  # fixed: same noise / flow times every pass
        for name, batches in val_sets.items():
            outs = [peval_step(train_state, b, jax.random.fold_in(rng, i)) for i, b in enumerate(batches)]
            for k in outs[0]:
                record[f"{name}/{k}"] = float(np.mean([float(o[k]) for o in outs]))
        record["val_time_s"] = time.time() - t0
        log(record)
        logging.info(
            f"[val] step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in record.items() if "/" in k)
        )

    start_step = int(train_state.step)
    train_start = time.time()
    prior_wall = 0.0
    if resuming:  # continue the wall-clock / GPU-hour count from the last logged record
        for line in (run_dir / "metrics.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec.get("step", -1) < start_step:
                prior_wall = rec.get("wall_clock_s", prior_wall)
    else:
        validate(0, 0.0)

    infos, step_times, waits = [], [], []
    t_last = time.time()
    for step in range(start_step, config.num_train_steps):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        t_wait = time.time()
        batch = next(data_iter)
        waits.append(time.time() - t_wait)

        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            jax.block_until_ready(info)
            now = time.time()
            n = len(infos)
            step_times.append((now - t_last) / n)
            t_last = now
            wall = prior_wall + now - train_start
            mean_info = {k: float(np.mean([float(i[k]) for i in infos])) for k in infos[0]}
            samples = (step + 1) * config.batch_size
            record = {
                "kind": "train",
                "step": step + 1,
                **mean_info,
                "lr": float(lr_schedule(step)),
                "samples_seen": samples,
                "epochs": samples / split["train_frames"],
                "step_time_s": step_times[-1],
                "data_wait_s": float(np.mean(waits)),
                "samples_per_s": config.batch_size / step_times[-1],
                "wall_clock_s": wall,
                "gpu_hours": wall / 3600 * n_gpus,
                "gpu_mem_peak_gib": peak_mem_gib(),
            }
            log(record)
            logging.info(
                f"step {step + 1}/{config.num_train_steps} loss={record['loss']:.4f} "
                f"grad_norm={record['grad_norm']:.3f} lr={record['lr']:.2e} "
                f"{record['step_time_s']:.2f}s/step (data wait {record['data_wait_s']:.2f}s) "
                f"eta {(config.num_train_steps - step - 1) * record['step_time_s'] / 3600:.1f}h"
            )
            infos, waits = [], []

        done = step + 1
        if done % args.val_interval == 0 or done == config.num_train_steps:
            validate(done, prior_wall + time.time() - train_start)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(ckpt_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    ckpt_manager.wait_until_finished()
    metrics_file.close()
    logging.info(f"done: {run_dir}")


if __name__ == "__main__":
    main()
