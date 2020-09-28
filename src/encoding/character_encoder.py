"""A character one-hot encoder."""

from typing import Dict, List

import numpy as np

from src.settings import symbols
from src.settings import paths
from src.helper.pickle import load_object
from src.helper.data_structures import sort_dict_by_value, revert_dictionary


class CharacterEncoder:
    def __init__(self, encoder_dict: Dict[str, int]):
        self.encoder = encoder_dict
        self.decoder = revert_dictionary(encoder_dict)

    def encode_char(self, char: str) -> int:
        return self.encoder[char] if char in self.encoder else self.encoder[symbols.UNKNOWN]

    def encode_sequence(self, sequence: str) -> np.ndarray:
        encoded = np.zeros(len(sequence) + 2, dtype=int)
        encoded[0] = self.encoder[symbols.SOS]
        for i, char in enumerate(sequence):
            encoded[i + 1] = self.encode_char(char)
        encoded[-1] = self.encoder[symbols.EOS]
        return encoded

    def dim(self):
        return len(self.encoder)

    def decode_label(self, label: int):
        return self.decoder[label]

    def decode_sequence(self, labels: List[int]):
        return ''.join(self.decode_label(label) for label in labels)


def get_encoder(n: int = 0) -> CharacterEncoder:
    if n > 0:
        frequencies = load_object(paths.CHARACTER_FREQUENCY_DICT)
        sorted_frequencies = sort_dict_by_value(frequencies)
        most_frequent_chars = [char for char, frequency in sorted_frequencies[:n]]
        code_symbols = most_frequent_chars + [symbols.SOS, symbols.EOS, symbols.UNKNOWN]
        encoder = {symbol: index for index, symbol in enumerate(code_symbols)}
    else:
        encoder = load_object(paths.WIKI_ENCODER_DICT)
    return CharacterEncoder(encoder)


def get_acl_encoder() -> CharacterEncoder:
    encoder_dict = load_object(paths.ACL_ENCODER_DICT)
    return CharacterEncoder(encoder_dict)
