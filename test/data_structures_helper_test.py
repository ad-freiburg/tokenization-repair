import unittest

from src.helper.data_structures import *


class DataStructuresHelperTest(unittest.TestCase):
    def test_gather(self):
        matrix = np.asarray([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],
                             [-1, -2, -3]])
        indices = [1, 0, 2, 1]
        self.assertEqual(tuple([2, 4, 9, -2]), tuple(gather(matrix, indices)))

    def test_argmax_and_max(self):
        argmax, value = argmax_and_max([1, 2, 5, -1, 3])
        self.assertEqual(2, argmax)
        self.assertEqual(5, value)

    def test_top_k_indices(self):
        vec = [1, 5, 3, 6, 8, -1]
        top_k = top_k_indices(vec, 3)
        self.assertEqual(3, len(top_k))
        self.assertTrue(1 in top_k)
        self.assertTrue(3 in top_k)
        self.assertTrue(4 in top_k)

    def test_top_k_indices_sorted(self):
        vec = [1, 5, 3, 6, 8, -1]
        top_k_sorted = top_k_indices_sorted(vec, 4)
        self.assertEqual((4, 3, 1, 2), tuple(top_k_sorted))

    def test_sorted_position(self):
        vec = [1, 5, 3, 6, 8, -1]
        self.assertEqual(0, sorted_position(vec, 4))
        self.assertEqual(1, sorted_position(vec, 3))
        self.assertEqual(2, sorted_position(vec, 1))
        self.assertEqual(3, sorted_position(vec, 2))
        self.assertEqual(4, sorted_position(vec, 0))
        self.assertEqual(5, sorted_position(vec, 5))

    def test_rank(self):
        frequencies = {'a': 100,
                       'b': 13,
                       'c': 10000,
                       'd': -1,
                       'e': 0}
        expected = {'c': 0,
                    'a': 1,
                    'b': 2,
                    'e': 3,
                    'd': 4}
        self.assertEqual(expected, frequency_rank(frequencies))

    def test_revert_dictionary(self):
        dict = {'a': 0,
                'b': -1,
                0: 'c'}
        expected = {0: 'a',
                    -1: 'b',
                    'c': 0}
        self.assertEqual(expected, revert_dictionary(dict))

    def test_unique_on_sorted(self):
        self.assertListEqual(
            [-1, 0, 1, 3, 7, 10],
            unique_on_sorted([-1, -1, 0, 1, 3, 3, 3, 7, 7, 10, 10])
        )
        self.assertListEqual(
            [1.0, 0.99, 0.5, 0],
            unique_on_sorted([1.0, 1.0, 1.0, 0.99, 0.5, 0])
        )


if __name__ == "__main__":
    unittest.main()
