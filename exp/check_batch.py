"""Pull one batch through openpi's training pipeline for an Experiment 1 dataset and check it. Run from openpi/.

  python check_batch.py exp1/clean out.png

Prints image/state/action shapes and the normalized action statistics (with pi05_libero's norm stats these should be
roughly mean 0, std 1 if our data matches the base model's data), and saves the first sample's camera images so the
orientation can be checked by eye against what the policy sees at test time.
"""
import dataclasses
import sys

import numpy as np
from PIL import Image

import exp1_config  # noqa: F401
from openpi.training import config as _config
from openpi.training import data_loader as _data



def main():  # guarded: the data loader's spawned workers re-import this file
    repo, out = sys.argv[1], sys.argv[2]
    cfg = _config.get_config("pi05_libero_lora")
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=repo), batch_size=8)
    obs, act = next(iter(_data.create_data_loader(cfg, shuffle=True, num_batches=1)))
    for k, v in obs.images.items():
        print("image", k, v.shape, "range %.2f..%.2f" % (float(v.min()), float(v.max())))
    print("state", obs.state.shape, "actions", act.shape)
    a = np.asarray(act)[..., :7].reshape(-1, 7)
    print("normalized actions  mean", a.mean(0).round(2), " std", a.std(0).round(2))
    print("prompt tokens", None if obs.tokenized_prompt is None else obs.tokenized_prompt.shape)
    imgs = [((np.asarray(v[0]) + 1) * 127.5).clip(0, 255).astype(np.uint8) for v in obs.images.values()]
    Image.fromarray(np.concatenate(imgs, axis=1)).save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
