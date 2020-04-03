import unittest

from src.encoding.character_encoder import CharacterEncoder, get_encoder
from src.settings import symbols


class CharacterEncoderTest(unittest.TestCase):
    def test_get_encoder(self):
        encoder = get_encoder(200)
        self.assertEqual(203, encoder.dim())

    def test_encoder(self):
        encoder_dict = {'a': 1, 'b': 2, symbols.SOS: 3, symbols.EOS: 4, symbols.UNKNOWN: 5}
        encoder = CharacterEncoder(encoder_dict)
        # encode:
        self.assertEqual(1, encoder.encode_char('a'))
        self.assertEqual(2, encoder.encode_char('b'))
        self.assertEqual(5, encoder.encode_char('c'))
        self.assertListEqual([3, 1, 2, 5, 4], list(encoder.encode_sequence("abc")))
        # decode:
        self.assertEqual("SOSabUNKEOS", encoder.decode_sequence([3, 1, 2, 5, 4]))


if __name__ == "__main__":
    unittest.main()
