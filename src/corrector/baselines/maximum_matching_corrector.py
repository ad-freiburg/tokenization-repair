from typing import Tuple

from src.helper.pickle import load_object
from src.settings import paths
from src.helper.data_structures import sort_dict_by_value
from src.sequence.functions import get_space_positions_in_merged


class MaximumMatchingCorrector:
    def __init__(self, n=None):
        token_frequencies = load_object(paths.TOKEN_FREQUENCY_DICT)
        if n is None:
            self.tokens = set(token_frequencies)
        else:
            self.tokens = set(token for token, _ in sort_dict_by_value(token_frequencies)[:n])
        self.max_token_len = max(len(token) for token in self.tokens)

    def longest_match(self, sequence: str, start: int) -> Tuple[bool, int]:
        end = min(len(sequence), start + self.max_token_len)
        while end > start + 1:
            token = sequence[start:end]
            if token in self.tokens:
                return True, end
            end = end - 1
        return False, end

    def correct(self, sequence: str):
        space_positions = get_space_positions_in_merged(sequence)
        sequence = sequence.replace(' ', '')
        prediction = ""
        start = 0
        last_matched = False
        while start < len(sequence):
            matched, end = self.longest_match(sequence, start)
            token = sequence[start:end]
            if start > 0:
                if last_matched or matched or start in space_positions:
                    prediction += ' '
            prediction += token
            start = end
            last_matched = matched
        return prediction
