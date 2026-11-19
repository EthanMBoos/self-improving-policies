"""Robometer-4B as the grader: last-frame success probability per rollout, written into the sidecar.

Runs anywhere with the standard library; the model itself runs in Robometer's own environment through
grader/programs/robometer_grade.py as a subprocess (one load per call, so grade whole directories, not single episodes).

    from grader.robometer import grade_rollouts, score
    grade_rollouts(pathlib.Path("runs/x/round_1/rollouts"), cfg.grader)   # fills grader_* fields in episodes_extra.jsonl
    score(pathlib.Path("clip.mp4"), "put the middle black bowl on the plate", cfg.grader)   # one float

Fields written per episode: grader_name, grader_score (last-frame success probability), grader_label (score at or
above the configured threshold), grader_final_progress, grader_max_success_prob. The per-frame curves go to
<rollout_dir>/grader_curves.json for the false-positive gallery.

Orientation: the grader scores <id>/<camera>.mp4; the default camera is agentview_grader, Robometer's LIBERO training
orientation (see scripts/make_grader_videos.py). Command line, for comparing orientations without touching the sidecar:
    python3 -m grader.robometer runs/x/round_1/rollouts --config configs/rft/x.json --camera agentview --no-sidecar
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import time

from rft import sidecar
from rft.config import GraderSettings

REPO = pathlib.Path(__file__).resolve().parent.parent
# The program lives in its own folder: Python puts a script's directory first on sys.path, and a sibling named
# robometer.py (this file) would shadow the installed robometer package inside it.
GRADE_SCRIPT = REPO / "grader" / "programs" / "robometer_grade.py"


def _run(manifest_lines: list[dict], out_path: pathlib.Path, grader: GraderSettings, log_path: pathlib.Path) -> dict:
    manifest = out_path.with_suffix(".manifest.jsonl")
    manifest.write_text("".join(json.dumps(m) + "\n" for m in manifest_lines))
    cmd = [
        str(REPO / grader.python), str(GRADE_SCRIPT),
        "--model", str((REPO / grader.model_dir).resolve()),
        "--max-frames", str(grader.max_frames), "--threshold", str(grader.threshold),
        "--batch-size", str(grader.batch_size), "--camera", grader.camera,
        "--manifest", str(manifest), "--out", str(out_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(out_path.read_text())


def grade_rollouts(rollout_dir: pathlib.Path, grader: GraderSettings, only_missing: bool = True) -> dict[str, dict]:
    """Grade every logged episode in `rollout_dir` (those without a score, by default). Returns the sidecar updates."""
    rollout_dir = pathlib.Path(rollout_dir)
    path = rollout_dir / sidecar.FILENAME
    records = sidecar.read_records(path)
    todo = [r for r in records if not r.get("error") and (not only_missing or r.get("grader_score") is None)]
    if not todo:
        return {}
    lines = []
    for r in todo:
        video = rollout_dir / r["dir"] / f"{grader.camera}.mp4"
        if not video.exists():
            raise FileNotFoundError(f"{video} missing; rollouts must be logged with video for grading")
        lines.append({"id": r["id"], "task": r["task"], grader.camera: str(video.resolve())})
    result = _run(lines, rollout_dir / "grader_out.json", grader, rollout_dir / "grader.log")
    timing = result.pop("__timing__", None)
    updates, curves = {}, {}
    for ep_id, res in result.items():
        s = float(res["score"])
        updates[ep_id] = {
            "grader_name": grader.name,
            "grader_score": s,
            "grader_label": int(s >= grader.threshold),
            "grader_final_progress": res.get("final_progress"),
            "grader_max_success_prob": res.get("max_success_prob"),
        }
        curves[ep_id] = {"progress": res.get("progress"), "success_probs": res.get("success_probs")}
    sidecar.update_records(path, updates)
    curves_path = rollout_dir / "grader_curves.json"
    existing = json.loads(curves_path.read_text()) if curves_path.exists() else {}
    existing.update(curves)
    curves_path.write_text(json.dumps(existing))
    if timing:
        with (rollout_dir / "grader_timing.jsonl").open("a") as f:
            f.write(json.dumps({"n": len(updates), **timing}) + "\n")
    return updates


def all_graded(rollout_dir: pathlib.Path) -> bool:
    records = sidecar.read_records(pathlib.Path(rollout_dir) / sidecar.FILENAME)
    return bool(records) and all(r.get("grader_score") is not None for r in records if not r.get("error"))


def score(video_path: pathlib.Path, instruction: str, grader: GraderSettings) -> float:
    """Last-frame success probability of one clip. Loads the model for a single call; use grade_rollouts for batches."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "out.json"
        result = _run([{"id": "one", "task": instruction, grader.camera: str(pathlib.Path(video_path).resolve())}],
                      out, grader, pathlib.Path(tmp) / "grader.log")
        return float(result["one"]["score"])


def main() -> None:
    import argparse

    from rft import metrics
    from rft.config import LoopConfig

    p = argparse.ArgumentParser(description="grade a rollout directory with Robometer")
    p.add_argument("rollout_dir", type=pathlib.Path)
    p.add_argument("--config", required=True, type=pathlib.Path)
    p.add_argument("--camera", default=None, help="override the config's camera (video name without .mp4)")
    p.add_argument("--model-dir", default=None, help="override the config's model_dir (e.g. checkpoints/Robometer-4B-fft-libero-infer)")
    p.add_argument("--tag", default=None, help="name for the output file in --no-sidecar mode (default: the camera)")
    p.add_argument("--no-sidecar", action="store_true", help="only report agreement; write grader_out.<camera>.json")
    p.add_argument("--force", action="store_true", help="re-grade episodes that already have a score")
    args = p.parse_args()
    grader = LoopConfig.load(args.config).grader
    if args.camera:
        grader.camera = args.camera
    if args.model_dir:
        grader.model_dir = args.model_dir
        grader.name = "robometer-" + pathlib.Path(args.model_dir).name.replace("-infer", "")
    rollout_dir = args.rollout_dir
    if not args.no_sidecar:
        updates = grade_rollouts(rollout_dir, grader, only_missing=not args.force)
        print(json.dumps({"graded": len(updates), "camera": grader.camera}))
        return
    records = [r for r in sidecar.read_records(rollout_dir / sidecar.FILENAME) if not r.get("error")]
    lines = [{"id": r["id"], "task": r["task"], grader.camera: str((rollout_dir / r["dir"] / f"{grader.camera}.mp4").resolve())}
             for r in records]
    for m in lines:
        if not pathlib.Path(m[grader.camera]).exists():
            raise FileNotFoundError(m[grader.camera])
    tag = args.tag or grader.camera
    result = _run(lines, rollout_dir / f"grader_out.{tag}.json", grader, rollout_dir / "grader.log")
    result.pop("__timing__", None)
    sim = [int(bool(r["sim_success"])) for r in records]
    scores = [float(result[r["id"]]["score"]) for r in records]
    sweep = metrics.threshold_sweep(sim, scores, sorted(set(grader.audit_thresholds) | {grader.threshold}))
    any_m = next(iter(sweep.values()))
    progress = [float(result[r["id"]].get("final_progress") or 0.0) for r in records]
    summary = {"model": grader.model_dir, "camera": grader.camera, "n": len(records), "auroc": any_m["auroc"],
               "auroc_final_progress": metrics.auroc(progress, sim), "brier": any_m["brier"],
               "thresholds": {t: {k: m[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall")} for t, m in sweep.items()}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
