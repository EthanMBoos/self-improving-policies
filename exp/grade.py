"""Score saved LIBERO episodes with one grader. Runs in the official Robometer env (torch + transformers).

  python grade.py --pool $EXP/pools/main --grader robometer --model robometer/Robometer-4B --view true --out scores/x.jsonl

Graders
  robometer     Robometer checkpoint via the official package. score = success prob at the last frame;
                also saves last-frame progress.
  roboreward    Qwen3-VL with RoboReward's rubric prompt (from the model card). score = expected 1-5 value from the
                answer-token probabilities; also saves the parsed text answer. Works for the RoboReward checkpoint
                and, as a control, for the base Qwen3-VL with the same prompt.
  yesno         Qwen3-VL asked "did the robot complete the task? yes or no". score = P(yes) / (P(yes) + P(no));
                also saves the text answer (whichever is more likely).

Views (raw robosuite frames are stored upside down)
  true     raw[::-1]        vertical flip only: the real camera image, and what Robometer's LIBERO loader trains on
  mirror   raw[::-1, ::-1]  180-degree rotation: what openpi feeds the policy (left-right mirror of true)

Camera: agent (third person), wrist, or both (side by side).
"""
import argparse
import json
import pathlib
import re
import time

import numpy as np
import torch

ROBOREWARD_PROMPT = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.
Task: {task}"""

YESNO_PROMPT = ("The video (or image of the final state) shows a robot attempting this task: \"{task}\". Look closely at the final frames. "
                "Did the robot fully complete the task? Answer with only one word: yes or no.")


def load_frames(ep_dir, view, camera, n):
    d = np.load(ep_dir / "episode.npz")
    def v(x):
        return x[:, ::-1] if view == "true" else x[:, ::-1, ::-1]
    if camera == "agent":
        frames = v(d["agentview"])
    elif camera == "wrist":
        frames = v(d["wrist"])
    else:
        frames = np.concatenate([v(d["agentview"]), v(d["wrist"])], axis=2)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int) if n > 1 else np.array([len(frames) - 1])
    return np.ascontiguousarray(frames[idx])


class RobometerGrader:
    def __init__(self, model):
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator
        self.device = torch.device("cuda")
        self.cfg, self.tok, proc, self.model = load_model_from_hf(model_path=model, device=self.device)
        self.model.eval()
        self.collate = setup_batch_collator(proc, self.tok, self.cfg, is_eval=True)
        loss = getattr(self.cfg, "loss", None)
        self.discrete = getattr(loss, "progress_loss_type", "l2").lower() == "discrete" if loss else False
        self.bins = (getattr(loss, "progress_discrete_bins", None) if loss else None) or \
            getattr(self.cfg.model, "progress_discrete_bins", 10)

    @torch.no_grad()
    def __call__(self, frames, task):
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs
        traj = Trajectory(frames=frames, frames_shape=tuple(frames.shape), task=task, id="0",
                          metadata={"subsequence_length": len(frames)}, video_embeddings=None)
        batch = self.collate([ProgressSample(trajectory=traj, sample_type="progress")])
        inputs = {k: (x.to(self.device) if hasattr(x, "to") else x) for k, x in batch["progress_inputs"].items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):  # the current unsloth patch mixes fp32 and bf16 otherwise
            r = compute_batch_outputs(self.model, self.tok, inputs, sample_type="progress",
                                      is_discrete_mode=self.discrete, num_bins=self.bins)
        prog = np.asarray(r["progress_pred"][0], dtype=float)
        succ = np.asarray(r.get("outputs_success", {}).get("success_probs", [[np.nan]])[0], dtype=float)
        return dict(score=float(succ[-1]), progress_last=float(prog[-1]), success_curve=succ.round(3).tolist(),
                    progress_curve=prog.round(3).tolist())


class QwenGrader:
    def __init__(self, model, mode):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.mode = mode
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=torch.bfloat16,
                                                                 device_map="cuda").eval()
        tok = self.proc.tokenizer
        def ids(words):
            out = set()
            for w in words:
                t = tok.encode(w, add_special_tokens=False)
                if len(t) == 1:
                    out.add(t[0])
            return sorted(out)
        self.yes, self.no = ids(["yes", "Yes", " yes", " Yes", "YES"]), ids(["no", "No", " no", " No", "NO"])
        self.digits = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 6)]

    def inputs(self, frames, prompt):
        if len(frames) <= 2:  # one or two frames: send them as images (Qwen3-VL video needs more frames)
            msgs = [{"role": "user", "content": [{"type": "image"}] * len(frames) + [{"type": "text", "text": prompt}]}]
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return self.proc(text=[text], images=list(frames), return_tensors="pt").to(self.model.device)
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        meta = dict(fps=2.0, total_num_frames=len(frames), frames_indices=list(range(len(frames))), duration=len(frames) / 2.0)
        x = self.proc(text=[text], videos=[list(frames)], video_metadata=[meta], do_sample_frames=False,
                      return_tensors="pt")
        return x.to(self.model.device)

    @torch.no_grad()
    def __call__(self, frames, task):
        if self.mode == "yesno":
            x = self.inputs(frames, YESNO_PROMPT.format(task=task))
            p = torch.softmax(self.model(**x).logits[0, -1].float(), -1)
            py, pn = p[self.yes].sum().item(), p[self.no].sum().item()
            return dict(score=py / max(py + pn, 1e-12), answer="yes" if py > pn else "no", mass=py + pn)
        x = self.inputs(frames, ROBOREWARD_PROMPT.format(task=task))
        g = self.model.generate(**x, max_new_tokens=32, do_sample=False, output_scores=True, return_dict_in_generate=True)
        new = g.sequences[0, x["input_ids"].shape[1]:].tolist()
        text = self.proc.tokenizer.decode(new, skip_special_tokens=True)
        m = re.search(r"ANSWER:\s*([1-5])", text)
        score = np.nan
        for step, t in enumerate(new):  # expected score at the first generated digit
            if t in self.digits:
                p = torch.softmax(g.scores[step][0].float(), -1)[self.digits]
                p = (p / p.sum()).cpu().numpy()
                score = float((p * np.arange(1, 6)).sum())
                break
        return dict(score=score, answer=int(m.group(1)) if m else None, text=text.strip()[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--grader", choices=["robometer", "roboreward", "yesno"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--view", choices=["true", "mirror"], default="true")
    ap.add_argument("--camera", choices=["agent", "wrist", "both"], default="agent")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--names", default=None, help="file with episode names to grade (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = pathlib.Path(a.pool)
    eps = sorted(p for p in (pool / "episodes").iterdir() if (p / "meta.json").exists())
    if a.names:
        keep = set(pathlib.Path(a.names).read_text().split())
        eps = [p for p in eps if p.name in keep]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["name"] for l in open(out)} if out.exists() else set()
    grader = RobometerGrader(a.model) if a.grader == "robometer" else QwenGrader(a.model, a.grader)
    t0 = time.time()
    for k, ep in enumerate(eps):
        if ep.name in done:
            continue
        meta = json.loads((ep / "meta.json").read_text())
        r = grader(load_frames(ep, a.view, a.camera, a.frames), meta["task"])
        r.update(name=ep.name, success=meta["success"], grader=a.grader, model=a.model, view=a.view,
                 camera=a.camera, frames=a.frames)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        if k % 25 == 0:
            print("%d/%d %.0fs %s" % (k, len(eps), time.time() - t0, {x: r[x] for x in ("name", "success", "score")}),
                  flush=True)


if __name__ == "__main__":
    main()
