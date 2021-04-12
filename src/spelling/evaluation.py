from typing import List, Tuple

from enum import Enum
import numpy as np

from src.edit_distance.edit_distance import levenshtein, get_operations, OperationTypes


class TokenErrorType(Enum):
    NONE = 0
    TOKENIZATION_ERROR = 1
    OCR_ERROR = 2
    MIXED = 3


def get_token_errors(original_text: str, operations: List[Tuple[OperationTypes, int, str]]) -> List[TokenErrorType]:
    labels = []
    space_deletions = {pos for op, pos, char in operations if op == OperationTypes.DELETE and char == ' '}
    space_insertions = {pos for op, pos, char in operations if op == OperationTypes.INSERT and char == ' '}
    n_spaces_at_pos = {i: 1 if char == ' ' else 0 for i, char in enumerate(original_text)}
    n_spaces_at_pos[len(original_text)] = 1
    for op, pos, char in operations:
        if char == ' ':
            if op == OperationTypes.INSERT:
                n_spaces_at_pos[pos] += 1
            else:
                n_spaces_at_pos[pos] -= 1
    other_changes = {pos for op, pos, char in operations if char != ' '}
    is_token_error = False
    is_ocr_error = False
    for i in range(len(original_text) + 1):
        if i in space_deletions or i in space_insertions:
            is_token_error = True
        if i in other_changes:
            is_ocr_error = True
        if i == len(original_text) or (original_text[i] == ' ' and i not in space_deletions) or i in space_insertions:
            for _ in range(n_spaces_at_pos[i]):
                if not is_token_error and not is_ocr_error:
                    labels.append(TokenErrorType.NONE)
                elif is_token_error and is_ocr_error:
                    labels.append(TokenErrorType.MIXED)
                elif is_token_error:
                    labels.append(TokenErrorType.TOKENIZATION_ERROR)
                else:
                    labels.append(TokenErrorType.OCR_ERROR)
            is_ocr_error = False
            if i not in space_insertions:
                is_token_error = False
    return labels


def longest_common_subsequence(seq1, seq2) -> List[Tuple[int, int]]:
    d = np.zeros(shape=(len(seq1) + 1, len(seq2) + 1), dtype=int)
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                d[i, j] = d[i - 1, j - 1] + 1
            else:
                d[i, j] = max(d[i - 1, j],
                              d[i, j - 1])
    i, j = len(seq1), len(seq2)
    matched = []
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            matched.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif d[i - 1, j] > d[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return matched[::-1]


def get_ground_truth_labels(correct_sequence: str, corrupt_sequence: str) -> List[TokenErrorType]:
    correct_tokens = correct_sequence.split()
    corrupt_tokens = corrupt_sequence.split()
    if corrupt_sequence != correct_sequence:
        d, matrix = levenshtein(corrupt_sequence, correct_sequence, return_matrix=True, substitutions=False)
        operations = get_operations(corrupt_sequence, correct_sequence, matrix, substitutions=False)
        error_types = get_token_errors(corrupt_sequence, operations)
    errors = [TokenErrorType.NONE for _ in correct_tokens]
    matched = longest_common_subsequence(correct_tokens, corrupt_tokens)
    no_error_tokens = {i for i, j in matched}
    for i in range(len(correct_tokens)):
        if i not in no_error_tokens:
            errors[i] = error_types[i]
    assert len(errors) == len(correct_tokens)
    return errors
