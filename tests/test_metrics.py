import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rft import metrics  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_wilson(self):
        self.assertEqual(metrics.wilson(0, 0), (0.0, 1.0))
        lo, hi = metrics.wilson(22, 50)  # the middle-bowl task at 50 trials: 31 to 58 percent
        self.assertAlmostEqual(lo, 0.31, delta=0.01)
        self.assertAlmostEqual(hi, 0.58, delta=0.01)
        lo, hi = metrics.wilson(30, 100)
        self.assertTrue(0.21 < lo < 0.23 and 0.39 < hi < 0.41)
        self.assertEqual(metrics.rate_with_ci(3, 10)["rate"], 0.3)

    def test_agreement_counts(self):
        sim = [1, 1, 0, 0, 1, 0]
        grader = [1, 0, 1, 0, 1, 0]
        a = metrics.agreement(sim, grader)
        self.assertEqual((a["tp"], a["fp"], a["fn"], a["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(a["precision"], 2 / 3)
        self.assertAlmostEqual(a["recall"], 2 / 3)
        self.assertAlmostEqual(a["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(a["success_rate_gap"], 0.0)
        self.assertIsNone(metrics.agreement([0, 0], [0, 0])["precision"])

    def test_score_metrics(self):
        sim = [1, 1, 0, 0]
        self.assertEqual(metrics.auroc([0.9, 0.8, 0.2, 0.1], sim), 1.0)
        self.assertEqual(metrics.auroc([0.5, 0.5, 0.5, 0.5], sim), 0.5)
        self.assertIsNone(metrics.auroc([0.1, 0.2], [1, 1]))
        a = metrics.agreement(sim, [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(a["brier"], (0.01 + 0.04 + 0.04 + 0.01) / 4)
        self.assertGreaterEqual(a["ece"], 0.0)
        self.assertAlmostEqual(a["score_mean_success"], 0.85)

    def test_threshold_sweep(self):
        sim = [1, 0, 1, 0]
        scores = [0.9, 0.7, 0.6, 0.1]
        sweep = metrics.threshold_sweep(sim, scores, [0.5, 0.65, 0.85])
        self.assertEqual(sweep["0.5"]["fp"], 1)
        self.assertEqual(sweep["0.65"]["fn"], 1)
        self.assertEqual(sweep["0.85"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
