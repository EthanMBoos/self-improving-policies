"""Robometer grader (run with Robometer's environment: third_party/robometer/.venv/bin/python).

Reads a manifest of episodes (JSONL: id, task, agentview video path) and writes {id: {label, score, ...}}:
    score = Robometer's success probability at the final frame (the soft reward)
    label = score > threshold (the binary reward)
plus the per-frame progress and success curves. Frames are subsampled uniformly to --max-frames (Robometer's
config uses 8), as in Robometer's own LIBERO reward wrapper.

Modes: one-shot (--manifest M --out O) or --server (see sip_rl/graders.py CommandGrader): the model stays in CPU
RAM between requests and is moved to the GPU only while grading.
"""

import argparse
import json
import pathlib
import sys
import time

import imageio.v3 as iio
import torch
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_utils import raw_dict_to_sample
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


class Robometer:
    def __init__(self, model_path: str, device: str):
        self.cfg, self.tok, proc, self.model = load_model_from_hf(model_path=model_path, device=torch.device(device))
        self.model.eval()
        loss_cfg = getattr(self.cfg, "loss", None)
        self.is_discrete = getattr(loss_cfg, "progress_loss_type", "l2").lower() == "discrete" if loss_cfg else False
        self.num_bins = getattr(loss_cfg, "progress_discrete_bins", None) or getattr(self.cfg.model, "progress_discrete_bins", 10)
        self.collate = setup_batch_collator(proc, self.tok, self.cfg, is_eval=True)

    def grade(self, eps, args) -> dict:
        out = {}
        for i in range(0, len(eps), args.batch_size):
            chunk = eps[i : i + args.batch_size]
            samples = [
                raw_dict_to_sample(
                    {"frames": iio.imread(e[args.camera]), "task": e["task"], "id": e["id"], "metadata": {},
                     "video_embeddings": None},
                    max_frames=args.max_frames, sample_type="progress",
                )
                for e in chunk
            ]
            batch = self.collate(samples)["progress_inputs"]
            batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.no_grad():
                r = compute_batch_outputs(self.model, self.tok, batch, sample_type="progress",
                                          is_discrete_mode=self.is_discrete, num_bins=self.num_bins)
            progress = r.get("progress_pred", [])
            success = (r.get("outputs_success") or {}).get("success_probs", [])
            for e, prog, succ in zip(chunk, progress, success):
                score = float(succ[-1])
                out[e["id"]] = {"label": int(score > args.threshold), "score": score,
                                "final_progress": float(prog[-1]),
                                "max_success_prob": float(max(succ)),
                                "progress": [float(x) for x in prog],
                                "success_probs": [float(x) for x in succ]}
        return out


def run_request(rm: Robometer, manifest: str, out_path: str, args, on_gpu: bool) -> None:
    eps = [json.loads(line) for line in open(manifest)]
    t0 = time.time()
    if not on_gpu:
        rm.model.to("cuda")
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    out = rm.grade(eps, args)
    t2 = time.time()
    peak = torch.cuda.max_memory_allocated() / 2**30
    if not on_gpu:
        rm.model.to("cpu")
        torch.cuda.empty_cache()
    out["__timing__"] = {"to_gpu_s": t1 - t0, "infer_s": t2 - t1, "to_cpu_s": time.time() - t2,
                         "episodes": len(eps), "peak_gpu_mem_gib": peak}
    pathlib.Path(out_path).write_text(json.dumps(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.add_argument("--server", action="store_true")
    p.add_argument("--model", required=True, help="absolute path of the (non-unsloth) Robometer checkpoint view")
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--camera", default="agentview")
    args = p.parse_args()

    if not args.server:
        t0 = time.time()
        rm = Robometer(args.model, "cuda")
        run_request(rm, args.manifest, args.out, args, on_gpu=True)
        timing = json.loads(pathlib.Path(args.out).read_text())
        timing["__timing__"]["load_s"] = time.time() - t0 - timing["__timing__"]["infer_s"]
        pathlib.Path(args.out).write_text(json.dumps(timing))
        return

    t0 = time.time()
    rm = Robometer(args.model, "cpu")
    print(f"loaded in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    print("__SIP_READY__", flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "exit":
            break
        run_request(rm, req["manifest"], req["out"], args, on_gpu=False)
        print("__SIP_DONE__", flush=True)


if __name__ == "__main__":
    main()
