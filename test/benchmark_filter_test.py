import unittest

from src.evaluation.functions import filter_prob_prediction_pairs
from src.sequence.corruption import CorruptionType, Corruption


class BenchmarkFilterTest(unittest.TestCase):
    def test_filter_prob_prediction_pairs(self):
        predictions_before = [
            (0.5, Corruption(CorruptionType.INSERTION, 127, ' ')),
            (0.9, Corruption(CorruptionType.DELETION, 128, ' ')),
            (0.8, Corruption(CorruptionType.INSERTION, 0, 'x'))
        ]
        expected_predictions = [
            (0.8, Corruption(CorruptionType.INSERTION, 0, 'x'))
        ]
        predictions_after = filter_prob_prediction_pairs(predictions_before)
        self.assertEqual(tuple(expected_predictions),
                         tuple(predictions_after))


if __name__ == "__main__":
    unittest.main()

