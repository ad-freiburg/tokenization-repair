from typing import List, Set

import numpy as np

from src.models.char_lm.character_language_model import CharacterLanguageModel


def score(probabilities, threshold):
    scores = (probabilities - threshold) / (1 - threshold)
    scores = np.maximum(scores, 0)
    return scores


def window_insertion(insertion_scores, deletion_scores, size):
    windowed_insertion_scores = np.zeros_like(insertion_scores)
    for i, p_ins in enumerate(insertion_scores):
        p_ins = insertion_scores[i]
        if p_ins == 0:
            continue
        left = max(0, i - size + 1)
        right = i + size
        window_max = left + np.argmax(insertion_scores[left:right])
        if i != window_max:
            continue
        left_del = max(0, i - size)
        right_del = i + size
        window_max_p_del = np.max(deletion_scores[left_del:right_del])
        if p_ins > window_max_p_del:
            windowed_insertion_scores[i] = p_ins
    return windowed_insertion_scores


def window_deletion(insertion_scores, deletion_scores, size):
    windowed_deletion_scores = np.zeros_like(deletion_scores)
    for i, p_del in enumerate(deletion_scores):
        if p_del == 0:
            continue
        left = max(0, i - size)
        right = i + size + 1
        window_max = left + np.argmax(deletion_scores[left:right])
        if i != window_max:
            continue
        left_ins = max(0, i - size + 1)
        right_ins = i + size + 1
        window_max_p_ins = np.max(insertion_scores[left_ins:right_ins])
        if p_del >= window_max_p_ins:
            windowed_deletion_scores[i] = p_del
    return windowed_deletion_scores


def window(insertion_scores, deletion_scores, size):
    window_insertion_scores = window_insertion(insertion_scores, deletion_scores, size)
    window_deletion_scores = window_deletion(insertion_scores, deletion_scores, size)
    return window_insertion_scores, window_deletion_scores


def apply_edits(sequence, insertion_positions, deletion_positions):
    edited_sequence = ""
    for i, char in enumerate(sequence):
        if i in insertion_positions:
            edited_sequence += ' '
        if i not in deletion_positions:
            edited_sequence += char
    return edited_sequence


def translate_positions_to_merged(sequence: str) -> List[int]:
    translated = []
    merged_pos = 0
    for char in sequence:
        translated.append(merged_pos)
        if char != ' ':
            merged_pos += 1
    if len(sequence) == 0 or sequence[-1] != ' ':
        translated.append(merged_pos)
    return translated


def get_space_positions(sequence: str) -> List[int]:
    return [i for i in range(len(sequence)) if sequence[i] == ' ']


class Sequence:
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.position_translation = translate_positions_to_merged(sequence)
        self.space_positions = {self.position_translation[i] for i in get_space_positions(sequence)}
        self.inserted_spaces = set()
        self.deleted_spaces = set()
        self.merged_sequence_length = len(sequence) - len(self.space_positions)

    def apply_operations(self, insertion_positions: Set[int], deletion_positions: Set[int]):
        new_sequence = ""
        new_position_translation = []
        new_space_positions = set()
        for i, char in enumerate(self.sequence):
            translated_position = self.position_translation[i]
            if i in insertion_positions:
                new_sequence += ' ' + char
                new_position_translation += [translated_position] * 2
                new_space_positions.add(translated_position)
                self.inserted_spaces.add(translated_position)
            elif i in deletion_positions:
                self.deleted_spaces.add(translated_position)
            else:
                new_sequence += char
                new_position_translation.append(translated_position)
                if char == ' ':
                    new_space_positions.add(translated_position)
        new_position_translation.append(self.merged_sequence_length)
        self.sequence = new_sequence
        self.position_translation = new_position_translation
        self.space_positions = new_space_positions

    def is_inserted(self, position: int) -> bool:
        return self.position_translation[position] in self.inserted_spaces

    def is_deleted(self, position: int) -> bool:
        return self.position_translation[position] in self.deleted_spaces


MAX_ITERATIONS = 30


class IterativeWindowCorrector:
    def __init__(self,
                 model: CharacterLanguageModel,
                 insertion_threshold: float,
                 deletion_threshold: float,
                 window_size: int,
                 verbose: bool = False):
        self.model = model
        self.space_index = self.model.get_encoder().encode_char(' ')
        self.insertion_threshold = insertion_threshold
        self.deletion_threshold = deletion_threshold
        self.window_size = window_size
        self.verbose = verbose

    def correct(self, sequence: str) -> str:
        sequence = Sequence(sequence)
        for iteration in range(MAX_ITERATIONS):
            prediction = self.model.predict(sequence.sequence)
            insertion_probs = prediction["insertion_probabilities"][:, self.space_index]
            deletion_probs = np.array([0 if sequence.sequence[i] != ' ' else p
                                       for i, p in enumerate(prediction["deletion_probabilities"])])

            insertion_scores = score(insertion_probs, self.insertion_threshold)
            insertion_scores = np.array([0 if sequence.is_inserted(i) else s
                                         for i, s in enumerate(insertion_scores)])

            deletion_scores = score(deletion_probs, self.deletion_threshold)
            deletion_scores = np.array([0 if sequence.is_deleted(i) else s
                                        for i, s in enumerate(deletion_scores)])

            insertion_scores, deletion_scores = window(insertion_scores, deletion_scores, self.window_size)

            insertion_positions = {i for i, p in enumerate(insertion_scores) if p > 0}
            deletion_positions = {i for i, p in enumerate(deletion_scores) if p > 0}

            if len(insertion_positions) == 0 and len(deletion_positions) == 0:
                break

            sequence.apply_operations(insertion_positions, deletion_positions)

            if self.verbose:
                print("inserts:", sorted(insertion_positions))
                print("deletes:", sorted(deletion_positions))
                print(sequence.sequence)

        return sequence.sequence
