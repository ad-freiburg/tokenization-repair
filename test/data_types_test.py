import unittest

from src.helper.data_types import is_float


class DataTypesTest(unittest.TestCase):
    def test_is_float(self):
        self.assertTrue(is_float("1"))
        self.assertTrue(is_float("-1"))
        self.assertTrue(is_float("1.23"))
        self.assertTrue(is_float("-1.23"))
        self.assertTrue(is_float("1e-1"))
        self.assertFalse(is_float("bla"))
        self.assertFalse(is_float("1,23"))
        self.assertFalse(is_float("None"))


if __name__ == "__main__":
    unittest.main()
