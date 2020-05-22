from typing import Tuple

from src.ngram.unigram_holder import UnigramHolder
from src.ngram.bigram_holder import BigramHolder


class BigramModel:
    def __init__(self, bigram_weight=0.5):
        self.unigrams = UnigramHolder()
        self.total_unigram_count = self.unigrams.total_count()
        self.bigrams = BigramHolder.load()
        self.bigram_weight = bigram_weight
        self.unigram_weight = 1 - bigram_weight

    def get_probability(self, bigram: Tuple[str, str]):
        p_unigram = self.get_unigram_probability(bigram[-1])
        if p_unigram == 0:
            return 0
        base_unigram_frequency = self.unigrams.get(bigram[0])
        if base_unigram_frequency > 0:
            bigram_frequency = self.bigrams.get(bigram)
            p_bigram = bigram_frequency / base_unigram_frequency
        else:
            p_bigram = 0
        p = self.unigram_weight * p_unigram + self.bigram_weight * p_bigram
        return p

    def get_unigram_probability(self, unigram: str) -> float:
        unigram_frequency = self.unigrams.get(unigram)
        p_unigram = unigram_frequency / self.total_unigram_count
        return p_unigram
