import unittest

from src.evaluation.sequence_result import count_occurences, corrupt_positions2predicted_positions
from src.sequence.corruption import CorruptionType, Corruption


class SequenceResultTest(unittest.TestCase):
    def test_count_occurences(self):
        values = [0, 1, 0, 3, 5, 7, 9]
        expected_occurences = [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        self.assertTrue((expected_occurences == count_occurences(10, values)).all())

    def test_corrupt_positions2predicted_positions(self):
        input = " Hellow orld."
        # predicted = "Hello world. "
        predictions = [
            Corruption(CorruptionType.INSERTION, 0, ' '),
            Corruption(CorruptionType.DELETION, 6, ' '),
            Corruption(CorruptionType.INSERTION, 7, ' '),
            Corruption(CorruptionType.DELETION, 13, ' ')
        ]
        expected_positions = [0,  # space
                              0,  # H
                              1,  # e
                              2,  # l
                              3,  # l
                              4,  # o
                              6,  # w
                              6,  # space
                              7,  # o
                              8,  # r
                              9,  # l
                              10, # d
                              11, # .
                              13] # end
        self.assertEqual(tuple(expected_positions),
                         tuple(corrupt_positions2predicted_positions(len(input),
                                                                     predictions)))


if __name__ == "__main__":
    unittest.main()
