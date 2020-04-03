import unittest

from src.data.simplewiki import Simplewiki


class SimplewikiTest(unittest.TestCase):
    def test_folds(self):
        f1 = Simplewiki.get_evaluation_files_shuffled(select_folds=[1])
        self.assertEqual(3334, len(f1))
        f2_f3 = Simplewiki.get_evaluation_files_shuffled(select_folds=[2, 3])
        self.assertEqual(6666, len(f2_f3))
        intersect = set(f1).intersection(set(f2_f3))
        self.assertEqual(0, len(intersect))
