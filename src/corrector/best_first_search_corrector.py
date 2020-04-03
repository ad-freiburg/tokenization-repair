import numpy as np
from heapq import heappush, heappop

from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.corrector.token_corrector import TokenCorrector
from src.helper.stochastic import log_likelihood
from src.sequence.corruption import Corruption, CorruptionType, revert_corruption


def get_score(sequence_log_likelihood, edit_operation_probability):
    # the smaller the better
    return -(sequence_log_likelihood + np.log(edit_operation_probability))


class CandidateQueue:
    def __init__(self):
        self.visited_sequences = set()
        self.sequence_log_likelihoods = dict()
        self.candidate_queue = []

    def pop(self):
        while True:
            score, sequence = heappop(self.candidate_queue)
            if sequence not in self.visited_sequences:
                self.visited_sequences.add(sequence)
                return sequence

    def push(self, sequence, score):
        heappush(self.candidate_queue, (score, sequence))

    def empty(self):
        while len(self.candidate_queue) > 0 and self.candidate_queue[0][1] in self.visited_sequences:
            heappop(self.candidate_queue)
        return len(self.candidate_queue) == 0


class BestFirstSearchCorrector(TokenCorrector):
    def __init__(self,
                 language_model: CharacterLanguageModel,
                 patience_steps: int):
        super(BestFirstSearchCorrector, self).__init__(language_model)
        self.patience_steps = patience_steps

    def process_candidate(self, sequence, threshold=0.1):
        candidates = []
        probs = self._get_probabilities(sequence)

        sequence_log_likelihood = log_likelihood(probs["sequence"])
        insertion_probs = probs["insertion"]
        deletion_probs = probs["deletion"]

        for i in range(len(insertion_probs)):
            operation_prob = insertion_probs[i]
            if operation_prob > threshold:
                candidate_sequence = revert_corruption(sequence, Corruption(CorruptionType.DELETION, i, ' '))
                score = get_score(sequence_log_likelihood, operation_prob)
                candidates.append((score, candidate_sequence))

        for pos in range(len(deletion_probs)):
            operation_prob = deletion_probs[pos]
            if operation_prob > threshold:
                candidate_sequence = revert_corruption(sequence, Corruption(CorruptionType.INSERTION, pos, ' '))
                score = get_score(sequence_log_likelihood, operation_prob)
                candidates.append((score, candidate_sequence))

        return sequence_log_likelihood, candidates

    def correct(self, sequence):
        queue = CandidateQueue()
        queue.push(sequence, -np.inf)
        best_sequence = ""
        best_log_lokelihood = -np.inf
        steps_without_improvement = 0
        while not queue.empty() and steps_without_improvement < self.patience_steps:
            candidate = queue.pop()
            candidate_log_likelihood, candidates = self.process_candidate(candidate)
            print(candidate_log_likelihood, candidate)
            if candidate_log_likelihood > best_log_lokelihood:
                best_log_lokelihood = candidate_log_likelihood
                best_sequence = candidate
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            for score, candidate in candidates:
                queue.push(candidate, score)
        return best_sequence