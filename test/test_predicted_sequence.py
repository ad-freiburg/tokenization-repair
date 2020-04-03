import unittest

from src.sequence.predicted_sequence import PredictedSequence
from src.sequence.corruption import CorruptionType, Corruption


class TestPredictedSequence(unittest.TestCase):
    def test_insert(self):
        sequence = "abc"
        ps = PredictedSequence(sequence)
        sequence = ps.insert(1)
        self.assertEqual("a bc", sequence)
        sequence = ps.insert(3)
        self.assertEqual("a b c", sequence)
        sequence = ps.insert(0)
        self.assertEqual(" a b c", sequence)
        sequence = ps.insert(6)
        self.assertEqual(" a b c ", sequence)
        self.assertEqual(Corruption(CorruptionType.DELETION, 1, ' '), ps.predicted_corruptions[0])
        self.assertEqual(Corruption(CorruptionType.DELETION, 2, ' '), ps.predicted_corruptions[1])
        self.assertEqual(Corruption(CorruptionType.DELETION, 0, ' '), ps.predicted_corruptions[2])
        self.assertEqual(Corruption(CorruptionType.DELETION, 3, ' '), ps.predicted_corruptions[3])

    def test_delete(self):
        sequence = " a b c "
        ps = PredictedSequence(sequence)
        sequence = ps.delete(0)
        self.assertEqual("a b c ", sequence)
        sequence = ps.delete(3)
        self.assertEqual("a bc ", sequence)
        sequence = ps.delete(4)
        self.assertEqual("a bc", sequence)
        sequence = ps.delete(1)
        self.assertEqual("abc", sequence)
        self.assertEqual(Corruption(CorruptionType.INSERTION, 0, ' '), ps.predicted_corruptions[0])
        self.assertEqual(Corruption(CorruptionType.INSERTION, 4, ' '), ps.predicted_corruptions[1])
        self.assertEqual(Corruption(CorruptionType.INSERTION, 6, ' '), ps.predicted_corruptions[2])
        self.assertEqual(Corruption(CorruptionType.INSERTION, 2, ' '), ps.predicted_corruptions[3])

    def test_filter(self):
        ps = PredictedSequence("")
        ps._inserted_original_positions = {5, 10, 15}
        ps._deleted_original_positions = {3, 10, 17}
        ps.predicted_corruptions = [
            Corruption(CorruptionType.DELETION, 5, ' '),
            Corruption(CorruptionType.DELETION, 10, ' '),
            Corruption(CorruptionType.DELETION, 15, ' '),
            Corruption(CorruptionType.INSERTION, 3, ' '),
            Corruption(CorruptionType.INSERTION, 10, ' '),
            Corruption(CorruptionType.INSERTION, 17, ' ')
        ]
        ps.probabilities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        filtered = ps._filter_inverse_predictions(list(zip(ps.probabilities, ps.predicted_corruptions)))
        self.assertEqual(
            ((0.9, Corruption(CorruptionType.DELETION, 5, ' ')),
             (0.7, Corruption(CorruptionType.DELETION, 15, ' ')),
             (0.6, Corruption(CorruptionType.INSERTION, 3, ' ')),
             (0.4, Corruption(CorruptionType.INSERTION, 17, ' '))),
            tuple(filtered)
        )


if __name__ == "__main__":
    unittest.main()
