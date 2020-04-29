from typing import Dict, Set, Tuple

from src.helper.pickle import load_object, dump_object
from src.settings import paths
from src.helper.files import file_exists
from src.ngram.unigram_holder import UnigramHolder
from src.ngram.bigram_holder import BigramHolder


def bounded_edit_distance(a: str, b: str) -> int:
    left_equal = right_equal = 0
    while left_equal < len(a) and left_equal < len(b):
        if a[left_equal] == b[left_equal]:
            left_equal += 1
        else:
            break
    a_remaining = len(a) - left_equal
    b_remaining = len(b) - left_equal
    while right_equal < a_remaining and right_equal < b_remaining:
        if a[-right_equal - 1] == b[-right_equal - 1]:
            right_equal += 1
        else:
            break
    a_remaining -= right_equal
    b_remaining -= right_equal
    if a_remaining == 2 and b_remaining == 2:
        if a[left_equal] == b[-right_equal - 1] and b[left_equal] == a[-right_equal - 1]:
            return 1
    diff = max(a_remaining, b_remaining)
    diff = min(2, diff)
    return diff


def get_stumps(token: str):
    return {token[:i] + token[(i + 1):] for i in range(len(token) + 1)}


MIN_TOKEN_FREQUENCY = 100


def get_stump_dict(unigrams: UnigramHolder) -> Dict[str, Set[str]]:
    if file_exists(paths.STUMP_DICT):
        return load_object(paths.STUMP_DICT)
    else:
        stump_dict = {}
        for token in unigrams.frequencies:
            if not token.isalpha():
                continue
            if unigrams.get(token) < MIN_TOKEN_FREQUENCY:
                continue
            for stump in get_stumps(token):
                if stump not in stump_dict:
                    stump_dict[stump] = {token}
                else:
                    stump_dict[stump].add(token)
        dump_object(stump_dict, paths.STUMP_DICT)
    return stump_dict


def query_stump_dict(stump_dict: Dict[str, Set[str]], query: str) -> Set[str]:
    result = set()
    for stump in get_stumps(query):
        if stump in stump_dict:
            for token in stump_dict[stump]:
                if bounded_edit_distance(query, token) <= 1:
                    result.add(token)
    return result


class FuzzyMatcher:
    def __init__(self,
                 unigrams: UnigramHolder,
                 bigrams: BigramHolder,
                 penalty: float):
        self.unigrams = unigrams
        self.stump_dict = get_stump_dict(unigrams)
        self.bigrams = bigrams
        self.penalty = penalty

    def match(self, query: str) -> Set[str]:
        return query_stump_dict(self.stump_dict, query)

    def fuzzy_unigram_frequency(self, query: str) -> Tuple[str, int]:
        best = query
        best_frequency = 0
        for token in self.match(query):
            frequency = self.unigrams.get(token)
            if token != query:
                frequency *= self.penalty
            if frequency > best_frequency:
                best, best_frequency = token, frequency
        return best, best_frequency

    def fuzzy_bigram_frequency(self, a: str, b: str, lower_bound: int = 0) -> Tuple[Tuple[str, str], int]:
        best_bigram = (a, b)
        best_frequency = 0
        a_matches = self.match(a)
        a_matches = {match for match in a_matches if self.unigrams.get(match) > lower_bound}
        a_ed = {match: 0 if match == a else 1 for match in a_matches}
        b_matches = self.match(b)
        b_matches = {match for match in b_matches if self.unigrams.get(match) > lower_bound}
        b_ed = {match: 0 if match == b else 1 for match in b_matches}
        for left in a_matches:
            for right in b_matches:
                bigram = (left, right)
                frequency = self.bigrams.get(bigram)
                if a_ed[left] > 0:
                    frequency *= self.penalty
                if b_ed[right] > 0:
                    frequency *= self.penalty
                if frequency > best_frequency:
                    best_bigram, best_frequency = bigram, frequency
        return best_bigram, best_frequency

    def best_fuzzy_split(self, token: str, lower_bound: int) -> Tuple[Tuple[str, str], Tuple[str, str], int]:
        best_frequency = 0
        best_split = token
        best_fuzzy_bigram = None
        for pos in range(1, len(token)):
            left, right = token[:pos], token[pos:]
            fuzzy_bigram, frequency = self.fuzzy_bigram_frequency(left, right, lower_bound)
            if frequency > best_frequency:
                best_split = left, right
                best_fuzzy_bigram = fuzzy_bigram
                best_frequency = frequency
        return best_split, best_fuzzy_bigram, best_frequency
