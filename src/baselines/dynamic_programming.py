from typing import List, Optional

from src.ngram.unigram_holder import UnigramHolder
from src.postprocessing.rule_based import RuleBasedPostprocessor
from src.postprocessing.bigram import BigramPostprocessor


MAX_WORD_LEN = 20
SINGLE_CHAR_WORDS = {'a', 'A', 'I'}


class DynamicProgrammingCandidate:
    def __init__(self,
                 tokens: List[str],
                 n_nonwords: int,
                 sorted_frequencies: List[int],
                 frequency_sum: int):
        self.tokens = tokens
        self.n_nonwords = n_nonwords
        self.sorted_frequencies = sorted_frequencies
        self.frequency_sum = frequency_sum

    def __str__(self):
        return "DPCandidate(%s, %i, %s %i)" % (str(self.tokens), self.n_nonwords, str(self.sorted_frequencies),
                                               self.frequency_sum)

    def __repr__(self):
        return str(self)


class DynamicProgrammingCorrector:
    def __init__(self,
                 bigram_postprocessing: bool,
                 allow_len_1: bool,
                 minimize_token_number: bool):
        self.unigrams = UnigramHolder()
        self.words = {word for word in self.unigrams.frequencies}
        if bigram_postprocessing:
            self.bigram_postprocessor = BigramPostprocessor(unigrams=self.unigrams)
        else:
            self.bigram_postprocessor = None
        self.allow_len_1 = allow_len_1
        self.minimize_token_number = minimize_token_number

    def locate_words(self, text: str) -> List[List[int]]:
        word_beginnings = [[] for _ in text]
        for i in range(len(text)):
            for j in range(i + 1, min(i + MAX_WORD_LEN, len(text)) + 1):
                word = text[i:j]
                if word in self.words:
                    word_beginnings[j - 1].append(i)
        return word_beginnings

    def is_word(self, token: str):
        if not self.allow_len_1:
            if len(token) == 1 and token not in SINGLE_CHAR_WORDS and token.isalpha():
                return False
        return token in self.words

    def candidate_comparison(self, candidate: DynamicProgrammingCandidate, other: DynamicProgrammingCandidate):
        # return self.frequency_sum > other.frequency_sum
        if self.minimize_token_number and len(candidate.tokens) != len(other.tokens):
            return len(candidate.tokens) < len(other.tokens)
        if candidate.n_nonwords != other.n_nonwords:
            return candidate.n_nonwords < other.n_nonwords
        for self_freq, other_freq in zip(candidate.sorted_frequencies, other.sorted_frequencies):
            if self_freq != other_freq:
                return self_freq > other_freq
        return len(candidate.tokens) < len(other.tokens)

    def _pick_best_candidate(self,
                             candidates: List[DynamicProgrammingCandidate]) -> Optional[DynamicProgrammingCandidate]:
        if len(candidates) == 0:
            return None
        best = candidates[0]
        for candidate in candidates[1:]:
            if self.candidate_comparison(candidate, best):
                best = candidate
        return best

    @staticmethod
    def _insert_into_sorted(array: List[int], x: int):
        i = 0
        while i < len(array):
            if x < array[i]:
                break
            i += 1
        return array[:i] + [x] + array[i:]

    def _new_candidate(self, token: str):
        is_word = self.is_word(token)
        frequency = self.unigrams.get(token)
        n_nonwords = 0 if is_word else 1
        frequencies = [frequency]
        tokens = [token]
        frequency_sum = frequency
        candidate = DynamicProgrammingCandidate(tokens, n_nonwords, frequencies, frequency_sum)
        return candidate

    def _expand_candidate(self, candidate: DynamicProgrammingCandidate, token: str):
        is_word = self.is_word(token)
        frequency = self.unigrams.get(token)
        n_nonwords = candidate.n_nonwords + (0 if is_word else 1)
        frequencies = self._insert_into_sorted(candidate.sorted_frequencies, frequency)
        tokens = candidate.tokens + [token]
        frequency_sum = candidate.frequency_sum + frequency
        candidate = DynamicProgrammingCandidate(tokens, n_nonwords, frequencies, frequency_sum)
        return candidate

    def correct(self, sequence: str) -> str:
        sequence = sequence.replace(' ', '')
        word_locations = self.locate_words(sequence)
        solutions = []
        for i in range(len(sequence)):
            candidates = []
            beginnings = word_locations[i]
            if len(beginnings) == 0:
                beginnings = [i]
            for b in beginnings:
                if b == 0:
                    token = sequence[:(i + 1)]
                    candidate = self._new_candidate(token)
                    candidates.append(candidate)
                elif solutions[b - 1] is not None:
                    previous = solutions[b - 1]
                    token = sequence[b:(i + 1)]
                    candidate = self._expand_candidate(previous, token)
                    candidates.append(candidate)
            solutions.append(self._pick_best_candidate(candidates))
        final_solution = solutions[-1]
        predicted = ' '.join(final_solution.tokens)
        if self.bigram_postprocessor is not None:
            predicted = self.bigram_postprocessor.correct(predicted)
        predicted = RuleBasedPostprocessor.correct(predicted)
        return predicted
