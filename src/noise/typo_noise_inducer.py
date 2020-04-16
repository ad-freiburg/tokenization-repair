from typing import List, Tuple

import random
from enum import Enum
import string

from src.noise.noise_inducer import NoiseInducer
from src.helper.stochastic import flip_coin


class SpellingCorruptionType(Enum):
    INSERTION = 0
    DELETION = 1
    REPLACEMENT = 2
    TRANSPOSITION = 3


INSERTION_CHARACTERS = string.ascii_lowercase


def all_possible_corruptions(token: str) -> List[Tuple[SpellingCorruptionType, int]]:
    corruptions = []
    # insertions:
    for pos in range(len(token) + 1):
        if pos == 0:
            if token[0].isalpha():
                corruptions.append((SpellingCorruptionType.INSERTION, pos))
        elif pos == len(token):
            if token[-1].isalpha():
                corruptions.append((SpellingCorruptionType.INSERTION, pos))
        else:
            if token[pos - 1].isalpha() or token[pos].isalpha():
                corruptions.append((SpellingCorruptionType.INSERTION, pos))
    # deletions:
    if len(token) > 1:
        for pos in range(len(token)):
            if token[pos].isalpha():
                corruptions.append((SpellingCorruptionType.DELETION, pos))
    # replacements:
    for pos in range(len(token)):
        if token[pos].isalpha():
            corruptions.append((SpellingCorruptionType.REPLACEMENT, pos))
    # transpositions:
    for pos in range(len(token) - 1):
        if token[pos].isalpha() and token[pos + 1].isalpha():
            corruptions.append((SpellingCorruptionType.TRANSPOSITION, pos))
    return corruptions


class TypoNoiseInducer(NoiseInducer):
    def __init__(self,
                 p: float,
                 seed: int):
        super(TypoNoiseInducer, self).__init__(seed)
        self.p = p

    def random_char(self):
        return self.rdm.choice(INSERTION_CHARACTERS)

    def apply_corruption(self,
                         token: str,
                         corruption: Tuple[SpellingCorruptionType, int]) -> str:
        corruption_type, position = corruption
        if corruption_type == SpellingCorruptionType.INSERTION:
            return token[:position] + self.random_char() + token[position:]
        if corruption_type == SpellingCorruptionType.DELETION:
            return token[:position] + token[(position + 1):]
        if corruption_type == SpellingCorruptionType.REPLACEMENT:
            return token[:position] + self.random_char() + token[(position + 1):]
        if corruption_type == SpellingCorruptionType.TRANSPOSITION:
            return token[:position] + token[position + 1] + token[position] + token[(position + 2):]

    def corrupt_token(self, token: str) -> str:
        if flip_coin(self.rdm, self.p):
            possible_corruptions = all_possible_corruptions(token)
            if len(possible_corruptions) > 0:
                corruption = self.rdm.choice(possible_corruptions)
                corrupt_token = self.apply_corruption(token, corruption)
                return corrupt_token
        return token

    def induce_noise(self, sequence: str) -> str:
        tokens = sequence.split(' ')
        corrupt_tokens = []
        for token in tokens:
            corrupt_token = self.corrupt_token(token)
            corrupt_tokens.append(corrupt_token)
        corrupt_sequence = ' '.join(corrupt_tokens)
        return corrupt_sequence
