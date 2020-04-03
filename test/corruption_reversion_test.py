import unittest

from src.sequence.corruption import CorruptionType, Corruption, revert_corruption


class CorruptionReversionTest(unittest.TestCase):
    def test_insertion_reversion(self):
        self.assertEqual("The algorithm runs in linear time.",
                         revert_corruption("The algo rithm runs in linear time.",
                                           Corruption(CorruptionType.INSERTION,
                                                      8,
                                                      ' ')))

    def test_deletion_reversion(self):
        self.assertEqual("The algorithm runs in linear time.",
                         revert_corruption("The algorithm runsin linear time.",
                                           Corruption(CorruptionType.DELETION,
                                                      18,
                                                      ' ')))


if __name__ == "__main__":
    unittest.main()
