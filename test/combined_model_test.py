import unittest
import numpy as np

from src.models.char_lm.combined_model import combine_probabilities, combine_compared


p1 = np.array([[0.5, 0.3, 0.2],
               [0.1, 0.1, 0.8],
               [0.3, 0.4, 0.3]])
p2 = np.array([[0.4, 0.3, 0.3],
               [0.2, 0.2, 0.6],
               [0.4, 0.2, 0.4]])
p_mean = np.array([[0.45, 0.3, 0.25],
                   [0.15, 0.15, 0.7],
                   [0.35, 0.3, 0.35]])
labels1 = [0, 1, 1]
labels2 = [0, 1, 2]
p_compare1 = [0.5, 0.1, 0.4]
p_compare2 = [0.4, 0.2, 0.4]
p_compare = [0.45, 0.15, 0.4]
p_compared = np.array([[0.5, 2/5, 5/14],
                       [0.5, 0.5, 14/17],
                       [7/15, 3/7, 7/15]])


class CombinedModelTest(unittest.TestCase):
    def test_combine_probabilities(self):
        p_combined = combine_probabilities(p1, p2, "mean")
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(p_mean[i, j], p_combined[i, j])

    def test_compared_probabilities(self):
        p_combined_compared = combine_compared(p1, p2, "mean", labels1, labels2)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(p_compared[i, j], p_combined_compared[i, j])


if __name__ == "__main__":
    unittest.main()
