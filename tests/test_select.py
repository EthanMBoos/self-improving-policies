"""Non-negotiable 1: the simulator's flag never influences selection under a VLM judge, and vice versa."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rft import select  # noqa: E402


def make_records(n=300, seed=0):
    rng = random.Random(seed)
    return [
        {"id": f"ep_{i:07d}", "sim_success": rng.random() < 0.4, "grader_score": rng.random(), "grader_name": "robometer"}
        for i in range(n)
    ]


class SelectTests(unittest.TestCase):
    def test_grader_judge_ignores_sim_success(self):
        records = make_records()
        before = select.select_episodes(records, "robometer", 0.65)
        flags = [r["sim_success"] for r in records]
        random.Random(1).shuffle(flags)
        for r, f in zip(records, flags):
            r["sim_success"] = f
        after = select.select_episodes(records, "robometer", 0.65)
        self.assertEqual(before, after)
        for r in records:
            del r["sim_success"]
        self.assertEqual(before, select.select_episodes(records, "robometer", 0.65))

    def test_sim_judge_ignores_grader(self):
        records = make_records()
        before = select.select_episodes(records, "sim")
        scores = [r["grader_score"] for r in records]
        random.Random(2).shuffle(scores)
        for r, s in zip(records, scores):
            r["grader_score"] = s
        self.assertEqual(before, select.select_episodes(records, "sim"))
        for r in records:
            del r["grader_score"]
            del r["grader_name"]
        self.assertEqual(before, select.select_episodes(records, "sim"))

    def test_threshold_is_inclusive_and_required(self):
        records = [{"id": "a", "grader_score": 0.65}, {"id": "b", "grader_score": 0.6499}]
        self.assertEqual(select.select_episodes(records, "robometer", 0.65), ["a"])
        with self.assertRaises(ValueError):
            select.select_episodes(records, "robometer")

    def test_missing_score_and_wrong_grader_raise(self):
        with self.assertRaises(ValueError):
            select.select_episodes([{"id": "a", "sim_success": True}], "robometer", 0.5)
        with self.assertRaises(ValueError):
            select.select_episodes([{"id": "a", "grader_score": 0.9, "grader_name": "other"}], "robometer", 0.5)

    def test_judged_rate(self):
        records = make_records(100)
        rate = select.judged_rate(records, "robometer", 0.5)
        self.assertAlmostEqual(rate, sum(r["grader_score"] >= 0.5 for r in records) / 100)
        self.assertIsNone(select.judged_rate([], "sim"))


if __name__ == "__main__":
    unittest.main()
