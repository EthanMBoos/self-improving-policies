"""scripts/loop.py selects with the grader named in the config, so a renamed checkpoint (the in-domain fallback) works."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rft import sidecar  # noqa: E402
from rft.config import LoopConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_loop():
    spec = importlib.util.spec_from_file_location("loop", REPO / "scripts" / "loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_round(root: Path, grader_name: str) -> Path:
    round_dir = root / "round_1"
    records = [
        {"id": "ep_0", "sim_success": True, "grader_name": grader_name, "grader_score": 0.9},
        {"id": "ep_1", "sim_success": False, "grader_name": grader_name, "grader_score": 0.7},
        {"id": "ep_2", "sim_success": True, "grader_name": grader_name, "grader_score": 0.2},
        {"id": "ep_3", "sim_success": False, "grader_name": grader_name, "grader_score": 0.1},
    ]
    sidecar.write_records(round_dir / "rollouts" / sidecar.FILENAME, records)
    return round_dir


class SelectKeptTests(unittest.TestCase):
    def setUp(self):
        self.loop = load_loop()
        self.base = json.loads((REPO / "configs" / "rft" / "middle_bowl.json").read_text())

    def config(self, grader_name: str) -> LoopConfig:
        d = dict(self.base, min_kept=1, grader=dict(self.base["grader"], name=grader_name,
                                                   model_dir="checkpoints/Robometer-4B-fft-libero-infer"))
        return LoopConfig.from_dict(d)

    def test_renamed_grader_in_config_selects_by_its_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = write_round(Path(tmp), "robometer-indomain")
            kept = self.loop.select_kept(self.config("robometer-indomain"), "robometer", round_dir)
        self.assertEqual(kept["ids"], ["ep_0", "ep_1"])
        self.assertEqual(kept["judge"], "robometer")
        self.assertEqual(kept["grader_name"], "robometer-indomain")
        self.assertEqual(kept["threshold"], 0.65)

    def test_scores_from_another_grader_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = write_round(Path(tmp), "robometer-indomain")
            with self.assertRaises(ValueError):
                self.loop.select_kept(self.config("robometer"), "robometer", round_dir)

    def test_sim_judge_ignores_grader_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = write_round(Path(tmp), "robometer-indomain")
            kept = self.loop.select_kept(self.config("robometer"), "sim", round_dir)
        self.assertEqual(kept["ids"], ["ep_0", "ep_2"])
        self.assertIsNone(kept["grader_name"])
        self.assertIsNone(kept["threshold"])


if __name__ == "__main__":
    unittest.main()
