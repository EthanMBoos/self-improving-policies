"""The first-grab instrument: grab segmentation, spot assignment, and the one-wrong-delivery flag."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import first_grasp as fg  # noqa: E402

SPOTS = fg.KITCHEN_SCENE2_SPOTS
TABLE_Z = 0.90
CARRY_Z = 1.00


def trace(*legs):
    """Build a trace from legs of (spot, closed, lifted, samples)."""
    eef, cmd = [], []
    for spot, closed, lifted, samples in legs:
        x, y = SPOTS[spot]
        for _ in range(samples):
            eef.append([x, y, CARRY_Z if lifted else TABLE_Z])
            cmd.append(1.0 if closed else -1.0)
    return {"eef": eef, "cmd": cmd, "finger": [0.0] * len(cmd)}


def carry(src, dst):
    """Close at src, lift, travel to dst closed, then open there."""
    return [(src, True, False, 2), (src, True, True, 2), (dst, True, True, 2), (dst, False, False, 3)]


class GrabTests(unittest.TestCase):
    def test_one_sample_blip_is_not_a_grab(self):
        t = trace(("middle", False, False, 3), ("middle", True, False, 1), ("middle", False, False, 3))
        self.assertEqual(fg.grabs(t), [])

    def test_grab_records_pick_release_and_lift(self):
        t = trace(("middle", False, False, 2), *carry("middle", "plate"))
        (g,) = fg.grabs(t)
        self.assertEqual(fg.nearest_spot(g["pick"], SPOTS), "middle")
        self.assertEqual(fg.nearest_spot(g["release"], SPOTS), "plate")
        self.assertTrue(g["released"])
        self.assertAlmostEqual(g["lift"], CARRY_Z - TABLE_Z)

    def test_grab_held_to_the_end_is_not_released(self):
        t = trace(("back", False, False, 2), ("back", True, False, 4))
        (g,) = fg.grabs(t)
        self.assertFalse(g["released"])


class DescribeTests(unittest.TestCase):
    def test_first_grab_is_the_first_spot_closed_on(self):
        t = trace(("back", False, False, 2), *carry("back", "plate"), *carry("middle", "plate"))
        d = fg.describe(t, SPOTS)
        self.assertEqual(d["first_grab"], "back")
        self.assertEqual(d["n_grabs"], 2)
        self.assertEqual(d["delivered"], ["back", "middle"])

    def test_one_wrong_delivery(self):
        only_back = fg.describe(trace(*carry("back", "plate")), SPOTS)
        self.assertTrue(fg.one_wrong_delivery(only_back, target="middle"))

    def test_right_bowl_is_not_a_wrong_delivery(self):
        only_middle = fg.describe(trace(*carry("middle", "plate")), SPOTS)
        self.assertFalse(fg.one_wrong_delivery(only_middle, target="middle"))

    def test_touching_another_bowl_is_not_one_delivery(self):
        back_then_front = fg.describe(trace(*carry("back", "plate"), ("front", True, False, 3), ("front", False, False, 2)), SPOTS)
        self.assertFalse(fg.one_wrong_delivery(back_then_front, target="middle"))

    def test_sliding_without_lifting_is_not_a_delivery(self):
        slid = [("back", True, False, 3), ("plate", True, False, 2), ("plate", False, False, 2)]
        self.assertEqual(fg.describe(trace(*slid), SPOTS)["delivered"], [])

    def test_no_grab(self):
        d = fg.describe(trace(("middle", False, False, 5)), SPOTS)
        self.assertEqual(d["first_grab"], "none")
        self.assertEqual(d["n_grabs"], 0)


if __name__ == "__main__":
    unittest.main()
