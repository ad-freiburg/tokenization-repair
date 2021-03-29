from typing import List, Tuple

from enum import Enum
import numpy as np
import json

import project
from src.helper.files import read_lines, write_file
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
    return matched


def get_ground_truth_labels(correct_sequence: str, corrupt_sequence: str) -> List[TokenErrorType]:
    correct_tokens = correct_sequence.split()
    corrupt_tokens = corrupt_sequence.split()
    if corrupt != correct_sequence:
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


if __name__ == "__main__":
    set = "development"
    start = 0
    end = 319

    benchmark_dir = "/home/hertel/tokenization-repair-dumps/data/spelling/ACL/" + set + "/"

    corrupt_paragraphs = read_lines(benchmark_dir + "corrupt.txt")
    spelling_paragraphs = read_lines(benchmark_dir + "spelling.txt")

    approaches = [
        "google",
        "the_one",
        "the_one+google",
        "gold+google",
        "hunspell",
        "the_one+hunspell",
        "gold+hunspell"
    ]

    predicted_sequences = {
        approach: read_lines(benchmark_dir + approach + ".txt") for approach in approaches
    }

    error_counts = {error: 0 for error in TokenErrorType}
    correct_prediction_counts = {approach: {error: 0 for error in TokenErrorType} for approach in approaches}
    sequences_data = []

    for i, (corrupt, spelling) in \
            enumerate(zip(corrupt_paragraphs, spelling_paragraphs)):
        if i < start or i > end:
            continue
        errors = get_ground_truth_labels(spelling, corrupt)
        for error in errors:
            error_counts[error] += 1
        correct_tokens = spelling.split()
        corrupt_tokens = corrupt.split()
        matched = longest_common_subsequence(correct_tokens, corrupt_tokens)
        matched_corrupt_tokens = {n for m, n in matched}
        corrupt_labels = ["NONE" if j in matched_corrupt_tokens else "WRONG" for j in range(len(corrupt_tokens))]
        sequence_data = {
            "corrupt": {
                "tokens": corrupt_tokens,
                "labels": corrupt_labels,
            },
            "correct": {
                "tokens": correct_tokens,
                "labels": [error.name for error in errors]
            },
            "predicted": {}
        }
        for approach in approaches:
            predicted = predicted_sequences[approach][i]
            predicted_tokens = predicted.split()
            matched = longest_common_subsequence(correct_tokens, predicted_tokens)
            labels = ["WRONG" for _ in predicted_tokens]
            for m, n in matched:
                correct_prediction_counts[approach][errors[m]] += 1
                labels[n] = errors[m]
            sequence_data["predicted"][approach] = {
                "tokens": predicted_tokens,
                "labels": [label.name if isinstance(label, TokenErrorType) else label for label in labels]
            }
        sequences_data.append(sequence_data)

    total_errors = sum(error_counts[error] for error in TokenErrorType if error != TokenErrorType.NONE)
    print(error_counts)
    for approach in approaches:
        print(correct_prediction_counts[approach])

    for approach in approaches:
        print("\n" + approach)
        counts = correct_prediction_counts[approach]
        for error_type in TokenErrorType:
            print(f"{error_type.name}: {counts[error_type]}/{error_counts[error_type]}")
        total_correct = sum(counts[error] for error in TokenErrorType if error != TokenErrorType.NONE)
        print(f"all: {total_correct}/{total_errors} ({total_correct / total_errors * 100:.2f}%)")

    data = {
        "total": {error.name: error_counts[error] for error in TokenErrorType},
        "correct": {approach: {error.name: correct_prediction_counts[approach][error] for error in TokenErrorType}
                    for approach in approaches},
        "sequences": sequences_data
    }
    write_file(benchmark_dir + "results.json", json.dumps(data))
