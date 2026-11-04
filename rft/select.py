"""Which rollouts enter the study pile. This is the only place a judge's verdict turns into training data.

Non-negotiable 1: with a VLM judge, the simulator's flag must never influence selection. Each judge reads exactly
one field: "sim" reads sim_success and nothing else; a grader judge reads grader_score and nothing else. The test
in tests/test_select.py permutes the other field and asserts the selection does not change.
"""

from __future__ import annotations

SIM_JUDGE = "sim"


def label(record: dict, judge: str, threshold: float | None = None) -> bool:
    if judge == SIM_JUDGE:
        return bool(record["sim_success"])
    if threshold is None:
        raise ValueError(f"judge {judge!r} needs a threshold")
    score = record.get("grader_score")
    if score is None:
        raise ValueError(f"episode {record.get('id')} has no grader_score; grade the rollouts first")
    name = record.get("grader_name")
    if name is not None and name != judge:
        raise ValueError(f"episode {record.get('id')} was graded by {name!r}, judge is {judge!r}")
    return float(score) >= threshold


def select_episodes(records: list[dict], judge: str, threshold: float | None = None) -> list[str]:
    """Ids of the episodes the judge approves, in the order given."""
    return [r["id"] for r in records if label(r, judge, threshold)]


def judged_rate(records: list[dict], judge: str, threshold: float | None = None) -> float | None:
    if not records:
        return None
    return sum(label(r, judge, threshold) for r in records) / len(records)
