"""Running several loops at once: a port per loop that reaches the stages, the final practice set, and the seed configs."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rft.config import LoopConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "rft"
BASE = CONFIGS / "middle_bowl.json"


def load_loop():
    spec = importlib.util.spec_from_file_location("loop", REPO / "scripts" / "loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MainTests(unittest.TestCase):
    """main() with the GPU stages replaced, in a run root inside the repo (the loop logs repo-relative paths)."""

    def setUp(self):
        self.loop = load_loop()
        self.loop.log = lambda msg: None
        tmp = tempfile.TemporaryDirectory(dir=REPO)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run_dir = self.root / "middle_bowl" / "robometer"
        self.calls = {"rollouts": [], "graded": []}
        self.loop.round_zero = lambda cfg, args, run_dir, graded: self.calls.update(config=args.config, port=cfg.port)
        self.loop.run_rollouts = lambda cfg, args, out, kind, k, ckpt, lerobot: self.calls["rollouts"].append((out, kind, k, ckpt, lerobot))
        self.loop.grade = lambda cfg, out: self.calls["graded"].append(out)

    def main(self, *extra, rounds_ok=True):
        self.loop.run_round = lambda cfg, args, run_dir, k, graded: rounds_ok
        argv = ["loop.py", "--config", str(BASE), "--judge", "robometer", "--run-root", str(self.root), *extra]
        with mock.patch.object(sys, "argv", argv):
            self.loop.main()

    def last_checkpoint(self, round_index=3):
        ckpt = self.run_dir / f"round_{round_index}" / "checkpoint" / "pi05_rft" / f"round_{round_index}" / "999"
        info = self.run_dir / f"round_{round_index}" / "checkpoint" / "train_info.json"
        info.parent.mkdir(parents=True)
        info.write_text(json.dumps({"final_checkpoint": str(ckpt)}))
        return ckpt

    def test_port_override_reaches_the_stages_through_the_saved_config(self):
        self.main("--port", "8005")
        self.assertEqual(self.calls["port"], 8005)
        self.assertEqual(self.calls["config"], self.run_dir / "config.json")
        self.assertEqual(LoopConfig.load(self.calls["config"]).port, 8005)

    def test_without_override_the_stages_read_the_given_config(self):
        self.main()
        self.assertEqual(self.calls["config"], BASE.resolve())
        self.assertEqual(self.calls["port"], LoopConfig.load(BASE).port)

    def test_final_practice_uses_the_last_checkpoint_and_the_next_rounds_seeds(self):
        ckpt = self.last_checkpoint()
        self.main("--final-practice")
        self.assertEqual(self.calls["rollouts"], [(self.run_dir / "round_4" / "rollouts", "rollout", 4, ckpt, None)])
        self.assertEqual(self.calls["graded"], [self.run_dir / "round_4" / "rollouts"])

    def test_no_final_practice_unless_asked(self):
        self.last_checkpoint()
        self.main()
        self.assertEqual(self.calls["rollouts"], [])

    def test_no_final_practice_when_a_round_stopped(self):
        self.last_checkpoint()
        self.main("--final-practice", rounds_ok=False)
        self.assertEqual(self.calls["rollouts"], [])

    def test_final_practice_seeds_must_stay_below_the_exam_seeds(self):
        # Rounds 1 to 3 fit below the exam seeds (the config validates); a fourth set would start on them.
        crowded = dict(json.loads(BASE.read_text()), rollout_seed_base=96000)
        path = self.root / "crowded.json"
        path.write_text(json.dumps(crowded))
        self.loop.run_round = lambda cfg, args, run_dir, k, graded: True
        argv = ["loop.py", "--config", str(path), "--judge", "robometer", "--run-root", str(self.root), "--final-practice"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
            self.loop.main()


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def changed_keys(name: str) -> set:
    base, other = flat(json.loads(BASE.read_text())), flat(json.loads((CONFIGS / name).read_text()))
    return {k for k in base.keys() | other.keys() if base.get(k) != other.get(k)}


class SeedConfigTests(unittest.TestCase):
    def test_seed_configs_change_only_the_seeds(self):
        for name in ("middle_bowl_s2.json", "middle_bowl_s3.json"):
            self.assertEqual(changed_keys(name), {"run_name", "rollout_seed_base", "train.seed"}, name)

    def test_indomain_config_changes_only_the_grader(self):
        self.assertEqual(changed_keys("middle_bowl_indomain.json"), {"run_name", "grader.name", "grader.model_dir"})

    def test_practice_layouts_never_repeat_across_seeds_or_reach_the_exam(self):
        seen = {}
        for name in ("middle_bowl.json", "middle_bowl_s2.json", "middle_bowl_s3.json"):
            cfg = LoopConfig.load(CONFIGS / name)
            for k in range(1, cfg.rounds + 2):  # the rounds plus the final practice set
                for seed in cfg.rollout_seeds(k):
                    self.assertNotIn(seed, seen, f"{name} round {k} repeats a layout of {seen.get(seed)}")
                    self.assertLess(seed, cfg.exam_seed_base)
                    seen[seed] = name

    def test_every_seed_takes_the_same_exam(self):
        exams = {tuple(LoopConfig.load(CONFIGS / n).exam_seeds())
                 for n in ("middle_bowl.json", "middle_bowl_s2.json", "middle_bowl_s3.json", "middle_bowl_indomain.json")}
        self.assertEqual(len(exams), 1)


if __name__ == "__main__":
    unittest.main()
