from typing import Tuple, List, Optional

from enum import Enum

import random
import string

from src.noise.noise_inducer import NoiseInducer


INSERT_CHARACTERS = string.ascii_letters


class TypoType(Enum):
    INSERTION = 0
    DELETION = 1
    REPLACEMENT = 2
    TRANSPOSITION = 3


TYPO_TYPES = list(TypoType)


class TokenTypoInducer(NoiseInducer):
    def __init__(self, p: float, seed: int):
        super().__init__(seed)
        self.p = p

    def flip_coin(self):
        return self.rdm.uniform(0, 1) < self.p

    def random_typo_type(self):
        return self.rdm.choice(TYPO_TYPES)

    def random_position(self, max: int) -> int:
        return self.rdm.randint(0, max)

    def random_char(self, except_char: Optional[str] = None) -> str:
        char = self.rdm.choice(INSERT_CHARACTERS)
        if except_char is not None:
            while char == except_char:
                char = self.rdm.choice(INSERT_CHARACTERS)
        return char

    def insert(self, token: str) -> str:
        pos = self.random_position(len(token))
        char = self.random_char()
        return token[:pos] + char + token[pos:]

    def delete(self, token: str) -> str:
        pos = self.random_position(len(token) - 1)
        return token[:pos] + token[(pos + 1):]

    def replace(self, token) -> str:
        pos = self.random_position(len(token) - 1)
        char = self.random_char(except_char=token[pos])
        return token[:pos] + char + token[(pos + 1):]

    def transpose(self, token) -> str:
        pos = self.random_position(len(token) - 2)
        return token[:pos] + token[pos + 1] + token[pos] + token[(pos + 2):]

    def corrupt_token(self, token: str) -> str:
        typo_type = self.random_typo_type()
        if typo_type == TypoType.INSERTION:
            corrupt = self.insert(token)
        elif typo_type == TypoType.DELETION and len(token) > 1:
            corrupt = self.delete(token)
        elif typo_type == TypoType.REPLACEMENT:
            corrupt = self.replace(token)
        elif typo_type == TypoType.TRANSPOSITION and len(token) > 1:
            corrupt = self.transpose(token)
        else:
            corrupt = token
        return corrupt

    def corrupt(self, sequence) -> Tuple[str, List[int]]:
        tokens = sequence.split(' ')
        corrupt_sequence = ""
        mask = []
        for t_i, token in enumerate(tokens):
            if t_i > 0:
                corrupt_sequence += ' '
                mask.append(1)
            if self.flip_coin():
                corrupt_token = self.corrupt_token(token)
                corrupt_sequence += corrupt_token
                mask.extend([0] * len(corrupt_token))
            else:
                corrupt_sequence += token
                mask.extend([1] * len(token))
        return corrupt_sequence, mask

    def induce_noise(self, sequence: str) -> str:
        corrupt, mask = self.corrupt(sequence)
        return corrupt
