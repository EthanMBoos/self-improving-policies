import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import audit  # noqa: E402
from rft import sidecar  # noqa: E402
from rft.config import LoopConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_shipped_configs_load_and_seeds_are_disjoint(self):
        for name in ("middle_bowl", "moka_pot"):
            cfg = LoopConfig.load(REPO / "configs" / "rft" / f"{name}.json")
            seen = set()
            for k in range(1, cfg.rounds + 1):
                s = set(cfg.rollout_seeds(k))
                self.assertEqual(len(s), cfg.rollouts_per_round)
                self.assertFalse(s & seen)
                seen |= s
            self.assertFalse(seen & set(cfg.exam_seeds()))
            self.assertEqual(len(cfg.exam_seeds()), cfg.exam_episodes)
            self.assertEqual(cfg.grader.threshold, 0.65)
            self.assertIn(cfg.grader.threshold, cfg.grader.audit_thresholds)

    def test_validation(self):
        base = json.loads((REPO / "configs" / "rft" / "middle_bowl.json").read_text())
        bad = dict(base, rollout_seed_stride=100)
        with self.assertRaises(ValueError):
            LoopConfig.from_dict(bad)
        bad = dict(base, exam_seed_base=2500)
        with self.assertRaises(ValueError):
            LoopConfig.from_dict(bad)
        bad = dict(base, train=dict(base["train"], warmup_steps=500))
        with self.assertRaises(ValueError):
            LoopConfig.from_dict(bad)
        cfg = LoopConfig.from_dict(base)
        self.assertEqual(LoopConfig.from_dict(cfg.to_dict()).to_dict(), cfg.to_dict())


class SidecarTests(unittest.TestCase):
    def test_roundtrip_and_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / sidecar.FILENAME
            self.assertEqual(sidecar.read_records(path), [])
            sidecar.append_record(path, {"id": "ep_1", "sim_success": True})
            sidecar.append_record(path, {"id": "ep_2", "sim_success": False})
            sidecar.update_records(path, {"ep_2": {"grader_score": 0.9, "grader_name": "robometer"}})
            recs = sidecar.read_records(path)
            self.assertEqual(recs[1]["grader_score"], 0.9)
            self.assertNotIn("grader_score", recs[0])
            with self.assertRaises(KeyError):
                sidecar.update_records(path, {"ep_9": {}})


class AuditTests(unittest.TestCase):
    def records(self, precise):
        recs = []
        for i in range(100):
            sim = i % 2 == 0
            # imprecise grader: approves only odd indices, which are all simulator failures (precision 0)
            score = (0.9 if sim else 0.2) if precise else (0.9 if i % 4 == 1 else 0.2)
            recs.append({"id": f"ep_{i}", "seed": i, "dir": f"ep_{i}", "sim_success": sim, "grader_score": score,
                         "grader_name": "robometer", "n_steps": 100})
        return recs

    def test_floor_pass_and_fail(self):
        good = audit.audit(self.records(True), [0.5, 0.65, 0.85], 0.65, 0.5)
        self.assertTrue(good["verdict"]["passes_floor"])
        self.assertEqual(good["sweep"]["0.65"]["fp"], 0)
        bad = audit.audit(self.records(False), [0.5, 0.65, 0.85], 0.65, 0.5)
        self.assertFalse(bad["verdict"]["passes_floor"])
        self.assertGreater(len(bad["false_positives_top"]), 0)
        md = audit.to_markdown(bad)
        self.assertIn("FAIL", md)
        self.assertIn("| 0.65 |", md)
        with self.assertRaises(ValueError):
            audit.audit([{"id": "x", "sim_success": True}], [0.5], 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()
