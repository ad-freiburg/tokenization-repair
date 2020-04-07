import unittest

from src.ngram.bigram_holder import BigramHolder


class BigramHolderTest(unittest.TestCase):
    def test_increment(self):
        holder = BigramHolder()
        holder.increment(["a", "b"])
        holder.increment(["a", "b"])
        holder.increment(["b", "c"])
        self.assertEqual(2, holder.get(["a", "b"]))
        self.assertEqual(1, holder.get(["b", "c"]))
        self.assertEqual(0, holder.get(["a", "c"]))
        self.assertEqual(0, holder.get(["a", "d"]))
        self.assertEqual(3, len(holder.unigram_encoder))
        self.assertEqual(2, len(holder.bigram_counts))

    def test_decode(self):
        holder = BigramHolder()
        holder.encode_unigram("a")
        holder.encode_unigram("b")
        self.assertEqual(("b", "a"), holder.decode((1, 0)))


if __name__ == "__main__":
    unittest.main()
