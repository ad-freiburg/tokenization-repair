from typing import List

import numpy as np

from src.ngram.bigram_model import BigramModel
from src.postprocessing.rule_based import RuleBasedPostprocessor


MAX_WORD_LEN = 20
EPSILON = 1e-16


class Solution:
    def __init__(self, sequence: str, last_token: str, score: float):
        self.sequence = sequence
        self.last_token = last_token
        self.score = score

    def __str__(self):
        return str((self.sequence, self.score))

    def __repr__(self):
        return str(self)


class BigramDynamicCorrector:
    def __init__(self):
        self.model = BigramModel()
        self.rule_based_postprocessor = RuleBasedPostprocessor()

    def is_token(self, text) -> bool:
        return self.model.unigrams.is_unigram(text)

    def locate_words(self, text: str) -> List[List[str]]:
        located_words = [[] for _ in text]
        for i in range(len(text)):
            for j in range(i + 1, min(i + MAX_WORD_LEN, len(text)) + 1):
                word = text[i:j]
                if self.is_token(word) or len(word) == 1:
                    located_words[i].append(word)
        return located_words

    def correct(self, sequence: str) -> str:
        sequence = sequence.replace(' ', '')
        words_at_position = self.locate_words(sequence)
        solutions = [{} for _ in sequence]
        for position in range(len(sequence)):
            words = words_at_position[position]
            for word in words:
                end_pos = position + len(word) - 1
                if position == 0:
                    p = self.model.get_unigram_probability(word) + EPSILON
                    solutions[end_pos][word] = Solution(word, word, np.log(p))
                else:
                    for previous_word in solutions[position - 1]:
                        prefix_solution = solutions[position - 1][previous_word]
                        bigram = (prefix_solution.last_token, word)
                        p = self.model.get_probability(bigram) + EPSILON
                        score = prefix_solution.score + np.log(p)
                        if word not in solutions[end_pos] or score > solutions[end_pos][word].score:
                            solutions[end_pos][word] = Solution(prefix_solution.sequence + ' ' + word,
                                                                word,
                                                                score)
        predicted = sequence
        best_score = -np.inf
        for last_word in solutions[-1]:
            solution = solutions[-1][last_word]
            if solution.score > best_score:
                predicted = solution.sequence
                best_score = solution.score
        predicted = self.rule_based_postprocessor.correct(predicted)
        return predicted
