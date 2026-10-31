"""Grader-in-the-loop PPO (piRL / RLinf) on LIBERO: entry point. Launch through scripts/rl/run_rl.sh.

Same structure as RLinf's examples/embodiment/train_embodied_agent.py, with our worker classes and runner
(sip_rl/): episodes are recorded, graded after each rollout phase, and rewards relabeled before the PPO update.
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

from sip_rl.runner import GradedRunner
from sip_rl.workers import GradedActor, RecordingEnvWorker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="../../configs/rl", config_name="libero10_pi05_lora")
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = GradedActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=placement.get_strategy("actor")
    )
    rollout = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=placement.get_strategy("rollout")
    )
    env = RecordingEnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=placement.get_strategy("env")
    )
    runner = GradedRunner(cfg=cfg, actor=actor, rollout=rollout, env=env, reward=None)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
