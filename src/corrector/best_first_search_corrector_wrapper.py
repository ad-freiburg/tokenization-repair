import numpy as np
from heapq import heappush, heappop

from src.evaluation.samples import get_space_corruptions
from src.sequence.corruption import Corruption, CorruptionType, revert_corruption


def get_score(sequence_log_likelihood, edit_operation_probability):
    # the smaller the better
    return -(sequence_log_likelihood + np.log(edit_operation_probability))


class CandidateQueue:
    def __init__(self, corrector, score, tolerance_steps):
        self.corrector = corrector
        self.score = score
        self.visited_sequences = set()
        self.sequence_log_likelihoods = dict()
        self.candidate_queue = []
        self.tolerance_steps = tolerance_steps
        self.steps_without_improvement = 0
        self.best_sequence = ""
        self.best_sequence_log_likelihood = -np.inf

    def get_log_likelihood(self, sequence):
        if sequence in self.sequence_log_likelihoods:
            return self.sequence_log_likelihoods[sequence]
        log_likelihood = self.corrector.sequence_log_likelihood(sequence)
        self.sequence_log_likelihoods[sequence] = log_likelihood
        return log_likelihood

    def add_candidate(self, sequence, score):
        if sequence in self.visited_sequences:
            return
        if self.score == "equal":
            self.candidate_queue.append((score, sequence))
        else:
            heappush(self.candidate_queue, (score, sequence))

    def pop(self):
        while True:
            if self.score == "equal":
                score, sequence = self.candidate_queue[0]
                self.candidate_queue = self.candidate_queue[1:]
            else:
                score, sequence = heappop(self.candidate_queue)
            if sequence not in self.visited_sequences:
                self.visited_sequences.add(sequence)
                log_likelihood = self.get_log_likelihood(sequence)
                if log_likelihood > self.best_sequence_log_likelihood:
                    self.best_sequence = sequence
                    self.best_sequence_log_likelihood = log_likelihood
                    self.steps_without_improvement = 0
                else:
                    self.steps_without_improvement += 1
                return sequence

    def get_score(self, sequence_log_likelihood, operation_prob):
        if self.score == "logsum":
            return get_score(sequence_log_likelihood, operation_prob)
        elif self.score == "equal":
            return 0
        raise Exception()

    def get_candidates(self, sequence, threshold=0.1):
        candidates = []

        sequence_log_likelihood = self.get_log_likelihood(sequence)
        insertion_probs = self.corrector.get_blank_insertion_probabilities_oneshot(sequence)
        deletion_probs = self.corrector.get_blank_deletion_probabilities_oneshot(sequence)

        for i in range(len(insertion_probs)):
            operation_prob = insertion_probs[i]
            if operation_prob > threshold:
                candidate_sequence = revert_corruption(sequence, Corruption(CorruptionType.DELETION, i, ' '))
                score = self.get_score(sequence_log_likelihood, operation_prob)
                candidates.append((score, candidate_sequence))

        for pos in deletion_probs:
            operation_prob = deletion_probs[pos]
            if operation_prob > threshold:
                candidate_sequence = revert_corruption(sequence, Corruption(CorruptionType.INSERTION, pos, ' '))
                score = self.get_score(sequence_log_likelihood, operation_prob)
                candidates.append((score, candidate_sequence))

        return candidates

    def get_candidates_spelling(self, sequence, insert_threshold=0.878156, delete_threshold=0.999977):
        candidates = []

        sequence_log_likelihood = self.get_log_likelihood(sequence)
        insertion_probs = self.corrector.get_all_insertion_probabilities(sequence)
        deletion_probs = self.corrector.get_deletion_probabilities_oneshot(sequence)

        for i in range(insertion_probs.shape[0]):
            for c_i in range(insertion_probs.shape[1]):
                operation_prob = insertion_probs[i, c_i]
                if operation_prob > insert_threshold:
                    candidate_sequence = revert_corruption(sequence, Corruption(CorruptionType.DELETION,
                                                                                i,
                                                                                self.corrector.decoder_dict[c_i]))
                    score = get_score(sequence_log_likelihood, operation_prob)
                    candidates.append((score, candidate_sequence))

        for pos in range(len(sequence)):
            operation_prob = deletion_probs[pos]
            if operation_prob > delete_threshold:
                candidate_sequence = revert_corruption(sequence, Corruption(CorruptionType.INSERTION,
                                                                            pos,
                                                                            sequence[pos]))
                score = get_score(sequence_log_likelihood, operation_prob)
                candidates.append((score, candidate_sequence))

        return candidates

    def terminated(self):
        if self.steps_without_improvement == self.tolerance_steps:
            return True
        while len(self.candidate_queue) > 0 and self.candidate_queue[0][1] in self.visited_sequences:
            if self.score == "equal":
                self.candidate_queue = self.candidate_queue[1:]
            else:
                heappop(self.candidate_queue)
        return len(self.candidate_queue) == 0


class BestFirstSearchCorrectorWrapper:
    def __init__(self, corrector, spelling=False, score="logsum", tolerance_steps=20):
        self.corrector = corrector
        self.spelling = spelling
        self.score = score
        self.tolerance_steps = tolerance_steps

    def predict(self, sequence):
        q = CandidateQueue(self.corrector, self.score, tolerance_steps=self.tolerance_steps)
        log_likelihood = q.get_log_likelihood(sequence)
        q.add_candidate(sequence, log_likelihood)

        while not q.terminated():
            candidate = q.pop()
            log_likelihood = q.get_log_likelihood(candidate)
            print(log_likelihood, candidate)
            if self.spelling:
                candidates = q.get_candidates_spelling(candidate)
            else:
                candidates = q.get_candidates(candidate)
            for score, candidate in candidates:
                q.add_candidate(candidate, score)

        predictions = get_space_corruptions(q.best_sequence, sequence) if not self.spelling else []  # TODO spelling predictions
        dummy_probs = [1 for _ in predictions]
        print(q.best_sequence)
        return zip(dummy_probs, predictions), q.best_sequence
