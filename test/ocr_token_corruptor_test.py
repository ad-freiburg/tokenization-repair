import unittest

from src.sequence.ocr_token_corruptor import OCRTokenCorruptor


class OCRTokenCorruptorTest(unittest.TestCase):
    def test_corrupt(self):
        self.assertEqual("a b c", OCRTokenCorruptor.corrupt("abc"))


if __name__ == "__main__":
    unittest.main()

