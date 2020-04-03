import numpy as np
from enum import Enum

from project import src
from src.helper.function import prob2score
from src.sequence.corruption import CorruptionType, Corruption
from src.settings import symbols


class TerminationCriterion(Enum):
    ITERATION_LIMIT_REACHED = 0
    PROBABILITIES_BELOW_THRESHOLD = 1
    LOOP_DETECTED = 2


class PredictionDetails:
    def __init__(self, position, type, character, probability, iteration, prefix, suffix):
        self.position = position
        self.type = type
        self.character = character
        self.probability = probability
        self.iteration = iteration
        self.prefix = prefix
        self.suffix = suffix

    def __str__(self):
        return str((self.position,
                    str(self.type),
                    self.character,
                    self.probability,
                    self.iteration,
                    self.prefix,
                    self.suffix))

    def __repr__(self):
        return str(self)


def inverse_corruption_type(corruption_type):
    if corruption_type == CorruptionType.INSERTION:
        return CorruptionType.DELETION
    elif corruption_type == CorruptionType.DELETION:
        return CorruptionType.INSERTION
    else:
        raise NotImplementedError


def invert_corruption(corruption):
    return Corruption(inverse_corruption_type(corruption.type), corruption.position, corruption.character)


def matrix_argmax(m):
    index = np.argmax(m)
    row = index // m.shape[1]
    column = index % m.shape[1]
    return row, column


def _get_original_position(original_positions, i):
    seq_len = len(original_positions)
    if i < seq_len:
        return original_positions[i]
    elif len(original_positions) > 0:
        return original_positions[-1] + 1
    return 0


def _filter_probabilities(probabilities, original_positions, filter_positions):
    if isinstance(probabilities, list):
        seq_len = len(probabilities)
    else:
        seq_len = probabilities.shape[0]
    for i in range(seq_len):
        orig_pos = _get_original_position(original_positions, i)
        if orig_pos in filter_positions:
            probabilities[i] = 0
    return probabilities


def _filter_insertion_probs(probabilities, original_positions, deletions):
    deletion_positions = [deletion[0] for deletion in deletions]
    deleted_indices = {pos: [] for pos in deletion_positions}
    for deletion in deletions:
        deleted_indices[deletion[0]].append(deletion[1])
    for i in range(probabilities.shape[0]):
        orig_pos = _get_original_position(original_positions, i)
        if orig_pos - 1 in deleted_indices:  # -1 because character was removed left to original position
            for index in deleted_indices[orig_pos - 1]:
                probabilities[i, index] = 0
    return probabilities


def predict(sequence, insertion_corrector, deletion_corrector, tokenization=True, insert=True, delete=True,
            prevent_multiple_insertion=True, insertion_threshold=None, deletion_threshold=None,
            max_iterations=200, return_details=False, snippet_len=20, eval_sequence_probability=False):
    predictions = []
    prediction_details = []
    probabilities = []
    original_positions = list(range(len(sequence)))
    threshold = min(insertion_threshold, deletion_threshold)
    prob = 1
    if eval_sequence_probability:
        best_log_sequence_probability = insertion_corrector.sequence_log_likelihood(sequence)
        print("log_sequence_probability = %.4f" % best_log_sequence_probability)
        best_sequence = sequence
        best_predictions = []
        best_prediction_details = []
        best_probabilities = []
    insert_positions = []
    #loop_positions = []
    iteration = 1
    while True:
        if prob < threshold:
            termination_criterion = TerminationCriterion.PROBABILITIES_BELOW_THRESHOLD
            break
        insert_prob = -1
        delete_prob = -1
        if tokenization:
            if insert:
                insert_probabilities = insertion_corrector.get_blank_insertion_probabilities_oneshot(sequence)
                if prevent_multiple_insertion:
                    insert_probabilities = _filter_probabilities(insert_probabilities, original_positions,
                                                                      insert_positions)
                #insert_probabilities = self._filter_probabilities(insert_probabilities, original_positions,
                #                                                  loop_positions)
                insert_pos = np.argmax(insert_probabilities)
                insert_prob = insert_probabilities[insert_pos]
                insert_char = ' '
            if delete and len(sequence) > 0:
                delete_probs = deletion_corrector.get_deletion_probabilities_oneshot(sequence)
                for i in range(len(sequence)):
                    if sequence[i] != ' ':
                        delete_probs[i] = 0
                #delete_probs = self._filter_probabilities(delete_probs, original_positions, loop_positions)
                delete_pos = np.argmax(delete_probs)
                delete_prob = delete_probs[delete_pos]
                delete_char = ' '
        else:
            if insert:
                insert_probs, insert_chars = insertion_corrector.get_insertion_probabilities_oneshot(sequence)
                if prevent_multiple_insertion:
                    insert_probs = _filter_probabilities(insert_probs, original_positions, insert_positions)
                #insert_probs = self._filter_probabilities(insert_probs, original_positions, loop_positions)
                insert_pos = np.argmax(insert_probs)
                insert_prob = insert_probs[insert_pos]
                insert_char = insert_chars[insert_pos]
            if delete and len(sequence) > 0:
                delete_probs = deletion_corrector.get_deletion_probabilities_oneshot(sequence)
                #delete_probs = self._filter_probabilities(delete_probs, original_positions, loop_positions)
                delete_pos = np.argmax(delete_probs)
                delete_prob = delete_probs[delete_pos]
                delete_char = sequence[delete_pos]
        if insert_prob >= insertion_threshold or delete_prob >= deletion_threshold:
            insertion_score = prob2score(insert_prob, insertion_threshold)
            deletion_score = prob2score(delete_prob, deletion_threshold)
            if insertion_score >= deletion_score:
                sequence_pos = insert_pos
                orig_pos = _get_original_position(original_positions, insert_pos)
                prediction = Corruption(CorruptionType.DELETION, orig_pos, insert_char)
                sequence_insert_char = insert_char
                if insert_char in [symbols.UNKNOWN,
                                   symbols.SOS,
                                   symbols.EOS]:
                    sequence_insert_char = '\ufffd'
                sequence = sequence[:insert_pos] + sequence_insert_char + sequence[insert_pos:]
                original_positions = original_positions[:insert_pos] + [orig_pos] + original_positions[insert_pos:]
                insert_positions.append(orig_pos)
                prob = min(prob, insert_prob)
            else:
                sequence_pos = delete_pos
                orig_pos = original_positions[delete_pos]
                prediction = Corruption(CorruptionType.INSERTION, orig_pos, delete_char)
                sequence = sequence[:delete_pos] + sequence[(delete_pos + 1):]
                original_positions = original_positions[:delete_pos] + original_positions[(delete_pos + 1):]
                prob = min(prob, delete_prob)
            print(prob, prediction)
            print(sequence)
            inverse_prediction = invert_corruption(prediction)
            if inverse_prediction in predictions:
                print("LOOP DETECTED!")
                termination_criterion = TerminationCriterion.LOOP_DETECTED
                break
                #index = predictions.index(inverse_prediction)
                #predictions = predictions[:index] + predictions[(index + 1):]
                #probabilities = probabilities[:index] + probabilities[(index + 1):]
                #loop_positions.append(orig_pos)
            predictions.append(prediction)
            probabilities.append(prob)
            if eval_sequence_probability:
                log_sequence_probability = insertion_corrector.sequence_log_likelihood(sequence)
                print("log_sequence_probability = %.4f" % log_sequence_probability)
                if log_sequence_probability > best_log_sequence_probability:
                    print("new best sequence")
                    best_log_sequence_probability = log_sequence_probability
                    best_sequence = sequence
                    best_predictions = predictions.copy()
                    best_prediction_details = prediction_details.copy()
                    best_probabilities = probabilities.copy()
            if return_details:
                if prediction.type == CorruptionType.DELETION:
                    suffix = sequence[(sequence_pos + 1):(sequence_pos + snippet_len + 1)]
                else:
                    suffix = sequence[sequence_pos:(sequence_pos + snippet_len)]
                prediction_details.append(PredictionDetails(prediction.position,
                                                            prediction.type,
                                                            prediction.character,
                                                            max(insert_prob, delete_prob),
                                                            iteration,
                                                            sequence[max(0, sequence_pos - snippet_len):sequence_pos],
                                                            suffix))
            if iteration == max_iterations:
                termination_criterion = TerminationCriterion.ITERATION_LIMIT_REACHED
                break
            iteration += 1
        else:
            prob = -1
    if eval_sequence_probability:
        sequence = best_sequence
        predictions = best_predictions
        prediction_details = best_prediction_details
        probabilities = best_probabilities
    if return_details:
        return sequence, termination_criterion, prediction_details
    else:
        return zip(probabilities, predictions), sequence


class GreedySentenceCorrectorWrapper:
    def __init__(self, insertion_corrector, deletion_corrector, tokenization=True, insert=True, delete=True,
                 prevent_multiple_insertion=True, insertion_threshold=None, deletion_threshold=None,
                 eval_sequence_probability=False):
        self.insertion_corrector = insertion_corrector
        self.deletion_corrector = deletion_corrector
        self.tokenization = tokenization
        self.insert = insert
        self.delete = delete
        self.prevent_multiple_insertion = prevent_multiple_insertion
        self.insertion_threshold = insertion_threshold if insertion_threshold is not None else float("inf")
        self.deletion_threshold = deletion_threshold if deletion_threshold is not None else float("inf")
        self.eval_sequence_probability = eval_sequence_probability

    def predict(self, sequence, max_iterations=200, return_details=False, snippet_len=20):
        return predict(sequence,
                       self.insertion_corrector,
                       self.deletion_corrector,
                       self.tokenization,
                       self.insert,
                       self.delete,
                       self.prevent_multiple_insertion,
                       self.insertion_threshold,
                       self.deletion_threshold,
                       max_iterations,
                       return_details,
                       snippet_len,
                       eval_sequence_probability=self.eval_sequence_probability)

    def end(self):
        self.insertion_corrector.end()
        self.deletion_corrector.end()
