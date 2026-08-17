import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reliability_tiers import reliability_tier


class ReliabilityTiers(unittest.TestCase):
    def test_held_out_subject_13_is_highly_reliable(self):
        # S13 acc=0.976 in the real held-out eval (models/eval_deployed.py)
        self.assertEqual(reliability_tier(13), "Highly reliable")

    def test_held_out_subject_12_is_somewhat_reliable(self):
        # S12 acc=0.854
        self.assertEqual(reliability_tier(12), "Somewhat reliable")

    def test_training_subject_gets_disclaimer_not_a_tier(self):
        # Subjects 1-11 trained the deployed model (models/train_model.py:77) -
        # no fair held-out score exists for them.
        self.assertEqual(
            reliability_tier(5),
            "Not independently tested for this person",
        )

    def test_out_of_range_subject_raises(self):
        with self.assertRaises(ValueError):
            reliability_tier(99)


if __name__ == "__main__":
    unittest.main()
