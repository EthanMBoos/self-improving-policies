"""Robometer-4B as the grader: last-frame success probability per rollout, written into the sidecar.

Runs anywhere with the standard library; the model itself runs in Robometer's own environment through
grader/programs/robometer_grade.py as a subprocess (one load per call, so grade whole directories, not single episodes).

    from grader.robometer import grade_rollouts, score
    grade_rollouts(pathlib.Path("runs/x/round_1/rollouts"), cfg.grader)   # fills grader_* fields in episodes_extra.jsonl
    score(pathlib.Path("clip.mp4"), "put the middle black bowl on the plate", cfg.grader)   # one float

Fields written per episode: grader_name, grader_score (last-frame success probability), grader_label (score at or
above the configured threshold), grader_final_progress, grader_max_success_prob. The per-frame curves go to
<rollout_dir>/grader_curves.json for the false-positive gallery.
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
