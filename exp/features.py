"""Frozen image features for the probe experiment (Exp 2). Runs in the Robometer env.

  python features.py --pool $EXP/pools/main --encoder siglip --out $EXP/pools/main/features_siglip.npz

For each episode: pooled embeddings of the first and last frame from the third-person and wrist cameras (true view),
plus a text embedding of the task (SigLIP only). Saved as one npz with names and arrays:
  agent_first, agent_last, wrist_first, wrist_last: (N, D)    text: (N, D) (SigLIP)
"""
import argparse
import json
import pathlib

import numpy as np
import torch

ENCODERS = {"siglip": "google/siglip-so400m-patch14-224", "dinov2": "facebook/dinov2-base"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--encoder", choices=list(ENCODERS), default="siglip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    name = ENCODERS[a.encoder]
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    eps = sorted(p for p in (pathlib.Path(a.pool) / "episodes").iterdir() if (p / "meta.json").exists())
    names, feats, texts = [], {k: [] for k in ("agent_first", "agent_last", "wrist_first", "wrist_last")}, []

    @torch.no_grad()
    def embed(images):
        x = proc(images=images, return_tensors="pt")["pixel_values"].cuda().half()
        if a.encoder == "siglip":
            return model.get_image_features(pixel_values=x).float().cpu().numpy()
        return model(pixel_values=x).pooler_output.float().cpu().numpy()

    for ep in eps:
        meta = json.loads((ep / "meta.json").read_text())
        d = np.load(ep / "episode.npz")
        ag, wr = d["agentview"], d["wrist"]
        imgs = [ag[0, ::-1], ag[-1, ::-1], wr[0, ::-1], wr[-1, ::-1]]
        e = embed([np.ascontiguousarray(i) for i in imgs])
        for k, v in zip(feats, e):
            feats[k].append(v)
        names.append(ep.name)
        texts.append(meta["task"])
    out = {k: np.stack(v) for k, v in feats.items()}
    if a.encoder == "siglip":
        with torch.no_grad():
            n = model.config.text_config.max_position_embeddings  # SigLIP is trained on text padded to this length
            t = proc(text=texts, padding="max_length", truncation=True, max_length=n, return_tensors="pt")
            out["text"] = model.get_text_features(input_ids=t["input_ids"].cuda()).float().cpu().numpy()
    np.savez(a.out, names=np.array(names), **out)
    print("saved", a.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
