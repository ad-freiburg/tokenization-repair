import unittest

from src.sequence.corruption import CorruptionType, Corruption
from src.evaluation.samples import corrupt_linebreaks_to_spaces, get_space_corruptions


class EvaluationSamplesTest(unittest.TestCase):
    def test_corrupt_linebreaks_to_spaces(self):
        original = "This is a test sequence.\nAnd another line.\nAnd a third one."
        corrupt_linebreaks = "Th isis a t-\nestseq-\nuence.And another line.\nAnd a third one. "
        expected_ground_truth = "This is a test sequence. And another line.\nAnd a third one."
        expected_corrupt = "Th isis a t estseq uence.And another line.\nAnd a third one. "
        ground_truth, corrupt = corrupt_linebreaks_to_spaces(original, corrupt_linebreaks)
        self.assertEqual(expected_ground_truth, ground_truth)
        self.assertEqual(expected_corrupt, corrupt)

    def test_get_space_corruptions(self):
        original = "This is a test sequence. "
        corrupt = " This isa test seq uence."
        expected_corruptions = [
            Corruption(CorruptionType.INSERTION, 0, ' '),
            Corruption(CorruptionType.DELETION, 8, ' '),
            Corruption(CorruptionType.INSERTION, 18, ' '),
            Corruption(CorruptionType.DELETION, 25, ' ')
        ]
        corruptions = get_space_corruptions(original, corrupt)
        self.assertEqual(expected_corruptions, corruptions)


if __name__ == "__main__":
    unittest.main()
