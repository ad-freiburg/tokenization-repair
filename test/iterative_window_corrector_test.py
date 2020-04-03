import unittest

from src.corrector.iterative_window_corrector import translate_positions_to_merged, Sequence


class IterativeWindowCorrectorTest(unittest.TestCase):
    def test_translate_positions_to_merged(self):
        seq = "The cat eats fish."
        #      0123345667890012345
        #                   1
        expected = [0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15]
        self.assertListEqual(expected, translate_positions_to_merged(seq))

    def test_apply_operations(self):
        sequence = Sequence("The cat eats fish.")
        #                    012345678901234567
        #                    0         1
        # translates to: Thecateatsfish.
        #                012345678901234
        #                0         1
        sequence.apply_operations({10, 17}, {3, 12})
        new_sequence = "Thecat ea tsfish ."
        self.assertEqual(new_sequence, sequence.sequence)
        new_position_translation = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 14, 15]
        self.assertListEqual(new_position_translation, sequence.position_translation)
        new_space_positions = {6, 8, 14}
        self.assertEqual(new_space_positions, sequence.space_positions)
        self.assertEqual({3, 10}, sequence.deleted_spaces)
        self.assertEqual({8, 14}, sequence.inserted_spaces)


if __name__ == "__main__":
    unittest.main()
