"""The per-episode metadata file next to every rollout directory and LeRobot dataset: episodes_extra.jsonl.

One JSON object per line. Written by the rollout worker (simulator fields), extended by the grader (grader fields).
Keys:
    id, suite, task_id, task, seed, init_state_source, sim_success, n_steps, max_steps, wall_s, error, dir,
    policy_checkpoint, round, inference_calls, infer_ms_mean,
    grader_name, grader_score, grader_label, grader_final_progress   (absent until graded)
"""

from __future__ import annotations

import json
import pathlib

FILENAME = "episodes_extra.jsonl"
SIM_KEYS = ("id", "suite", "task_id", "task", "seed", "init_state_source", "sim_success", "n_steps", "max_steps")
GRADER_KEYS = ("grader_name", "grader_score", "grader_label", "grader_final_progress")


def read_records(path: str | pathlib.Path) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write_records(path: str | pathlib.Path, records: list[dict]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    tmp.replace(p)


def append_record(path: str | pathlib.Path, record: dict) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(record) + "\n")


def update_records(path: str | pathlib.Path, updates: dict[str, dict]) -> list[dict]:
    """Merge `updates[id]` into the record with that id; unknown ids raise."""
    records = read_records(path)
    by_id = {r["id"]: r for r in records}
    missing = set(updates) - set(by_id)
    if missing:
        raise KeyError(f"episodes not in {path}: {sorted(missing)[:5]}")
    for ep_id, fields in updates.items():
        by_id[ep_id].update(fields)
    write_records(path, records)
    return records
