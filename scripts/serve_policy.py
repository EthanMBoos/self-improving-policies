"""Serve a policy checkpoint over openpi's websocket protocol (port 8000) for the LIBERO eval client.

Run from the repo root with openpi's environment, e.g.:
    third_party/openpi/.venv/bin/python scripts/serve_policy.py --config pi05_libero --dir checkpoints/pi05_libero
    third_party/openpi/.venv/bin/python scripts/serve_policy.py --config pi05_base_zeroshot --dir checkpoints/pi05_base
    third_party/openpi/.venv/bin/python scripts/serve_policy.py --config pi05_libero_lora \
        --dir checkpoints/pi05_libero_lora/sft_5k/4999

Norm stats come from <dir>/assets when the checkpoint has them, otherwise from the official LIBERO stats.
"""

import argparse
import logging
import pathlib
import socket

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize

import sip_common


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, choices=sorted(sip_common.CONFIGS))
    p.add_argument("--dir", required=True, type=pathlib.Path, help="checkpoint directory (contains params/)")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    config = sip_common.get_config(args.config)
    ckpt = args.dir.resolve()
    norm_stats = None
    if not (ckpt / "assets" / sip_common.LIBERO_REPO_ID).exists():
        stats_dir = pathlib.Path(sip_common.LIBERO_ASSETS.assets_dir) / sip_common.LIBERO_REPO_ID
        logging.info(f"no norm stats in checkpoint, using {stats_dir}")
        norm_stats = normalize.load(stats_dir)
    logging.info(f"config {args.config}: {config.model}")
    policy = policy_config.create_trained_policy(config, ckpt, norm_stats=norm_stats)
    logging.info(f"Creating server (host: {socket.gethostname()})")
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
