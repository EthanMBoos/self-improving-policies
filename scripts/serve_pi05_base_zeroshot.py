"""Serve pi05_base zero-shot on LIBERO: base params + pi05_libero data pipeline & LIBERO norm stats.

Keeps the pi0.5 default discrete_state_input=True (how the base model was pretrained);
the pi05_libero config sets it to False for its fine-tune. pi05_base ships no LIBERO norm stats,
so we borrow the ones bundled with the pi05_libero checkpoint.

Run with openpi's environment (from the repo root):
    third_party/openpi/.venv/bin/python scripts/serve_pi05_base_zeroshot.py
"""

import dataclasses
import logging
import pathlib
import socket

from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as _config

CKPT = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"

logging.basicConfig(level=logging.INFO, force=True)
cfg = _config.get_config("pi05_libero")
cfg = dataclasses.replace(cfg, model=pi0_config.Pi0Config(pi05=True, action_horizon=10))
logging.info(f"model config: {cfg.model}")
norm_stats = normalize.load(CKPT / "pi05_libero/assets/physical-intelligence/libero")
policy = policy_config.create_trained_policy(cfg, CKPT / "pi05_base", norm_stats=norm_stats)
logging.info(f"Creating server (host: {socket.gethostname()})")
websocket_policy_server.WebsocketPolicyServer(
    policy=policy, host="0.0.0.0", port=8000, metadata=policy.metadata
).serve_forever()
