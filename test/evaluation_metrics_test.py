import unittest

from src.sequence.corruption import Corruption, CorruptionType
from src.evaluation.metrics import tp, fp, fn

CASE1_TRUTH = {
    Corruption(CorruptionType.INSERTION, 3, ' '),
    Corruption(CorruptionType.DELETION, 5, ' '),  # fn
    Corruption(CorruptionType.DELETION, 7, ' '),
    Corruption(CorruptionType.INSERTION, 9, ' '),  # fn
    Corruption(CorruptionType.DELETION, 12, ' '),
    Corruption(CorruptionType.INSERTION, 15, ' '),  # fn
    Corruption(CorruptionType.INSERTION, 99, ' ')  # fn
}

CASE1_PREDICTION = {
    Corruption(CorruptionType.INSERTION, 3, ' '),  # tp
    Corruption(CorruptionType.DELETION, 7, ' '),  # tp
    Corruption(CorruptionType.DELETION, 12, ' '),  # tp
    Corruption(CorruptionType.DELETION, 15, ' '),  # fp
    Corruption(CorruptionType.DELETION, 16, ' '),  # fp
    Corruption(CorruptionType.INSERTION, 18, ' '),  # fp
    Corruption(CorruptionType.DELETION, 21, ' '),  # fp
    Corruption(CorruptionType.INSERTION, 33, ' ')  # fp
}


class EvaluationMetricsTest(unittest.TestCase):
    def test_tp(self):
        expected_tp = {
            Corruption(CorruptionType.INSERTION, 3, ' '),  # tp
            Corruption(CorruptionType.DELETION, 7, ' '),  # tp
            Corruption(CorruptionType.DELETION, 12, ' '),  # tp
        }
        self.assertEqual(expected_tp, tp(CASE1_TRUTH, CASE1_PREDICTION))

    def test_fp(self):
        expected_fp = {
            Corruption(CorruptionType.DELETION, 15, ' '),  # fp
            Corruption(CorruptionType.DELETION, 16, ' '),  # fp
            Corruption(CorruptionType.INSERTION, 18, ' '),  # fp
            Corruption(CorruptionType.DELETION, 21, ' '),  # fp
            Corruption(CorruptionType.INSERTION, 33, ' ')  # fp
        }
        self.assertEqual(expected_fp, fp(CASE1_TRUTH, CASE1_PREDICTION))

    def test_fn(self):
        expected_fn = {
            Corruption(CorruptionType.DELETION, 5, ' '),  # fn
            Corruption(CorruptionType.INSERTION, 9, ' '),  # fn
            Corruption(CorruptionType.INSERTION, 15, ' '),  # fn
            Corruption(CorruptionType.INSERTION, 99, ' ')  # fn
        }
        self.assertEqual(expected_fn, fn(CASE1_TRUTH, CASE1_PREDICTION))


if __name__ == "__main__":
    unittest.main()
