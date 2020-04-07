from typing import List, Tuple, Iterable

from src.ngram.tokenizer import Tokenizer
from src.settings import paths
from src.helper.pickle import load_object, dump_object
from src.helper.data_structures import revert_dictionary


class BigramHolder:
    def __init__(self):
        self.unigram_encoder = {}
        self.bigram_counts = {}

    def __len__(self):
        return len(self.bigram_counts)

    def encode_unigram(self, unigram: str) -> int:
        if unigram not in self.unigram_encoder:
            self.unigram_encoder[unigram] = len(self.unigram_encoder)
        return self.unigram_encoder[unigram]

    def encode_bigram(self, bigram: Iterable[str]) -> Tuple[int, ...]:
        return tuple(self.encode_unigram(unigram) for unigram in bigram)

    def increment(self, bigram: List[str]):
        encoded = self.encode_bigram(bigram)
        if encoded not in self.bigram_counts:
            self.bigram_counts[encoded] = 1
        else:
            self.bigram_counts[encoded] += 1

    def get(self, bigram: Iterable[str]) -> int:
        for unigram in bigram:
            if unigram not in self.unigram_encoder:
                return 0
        encoded = self.encode_bigram(bigram)
        if encoded not in self.bigram_counts:
            return 0
        return self.bigram_counts[encoded]

    def save(self):
        dump_object(self, paths.BIGRAM_HOLDER)

    @staticmethod
    def load():
        return load_object(paths.BIGRAM_HOLDER)

    def decode(self, encoded: Tuple[int, int]) -> Tuple[str, str]:
        if not hasattr(self, "unigram_decoder"):
            self.unigram_decoder = revert_dictionary(self.unigram_encoder)
        return tuple(self.unigram_decoder[label] for label in encoded)
