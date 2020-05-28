"""
Module containing edit distance utility functions, that are used for input
generation and fixing evaluations.
"""
from constants import TYPO_ADD, TYPO_CHANGE, TYPO_DEL, TYPO_NOCHANGE

import numpy as np


def apply_operations(text, operations_sequence):
    """
    Apply edit operations on a given text.

    :param str text: Given text
    :param list operations_sequence: Sequence of edit operations
    :rtype: str
    :returns: Editted string
    """
    text = [c for c in text]
    for operation in operations_sequence[::-1]:
        idx = operation[0]
        typ = operation[1]
        if typ == TYPO_DEL:
            del text[idx]
        elif typ == TYPO_ADD:
            char = operation[2]
            text.insert(idx, char)
        else:
            assert typ == TYPO_NOCHANGE
    return ''.join(text)


def edit_operations(correct, non_correct):
    """
    Get edit operations from the correct text to non-correct text

    :param str correct: Correct text
    :param str non_correct: Non-correct text
    :rtype: list
    :returns: Sequence of edit operations
    """
    INF = len(correct) + len(non_correct) + 1
    edit_dist = [[INF for j in range(len(non_correct) + 1)]
                 for i in range(2)]
    if len(correct) * len(non_correct) > 5 * 10**6:
        operation = np.zeros((len(correct) + 1, len(non_correct) + 1),
                             dtype=np.int8)
    else:
        operation = [[None for j in range(len(non_correct) + 1)]
                     for i in range(len(correct) + 1)]
    for i in range(len(correct) + 1):
        i2 = i & 1
        edit_dist[i2] = [INF for _ in range(len(non_correct) + 1)]
        for j in range(len(non_correct) + 1):
            if i == 0 and j == 0:
                edit_dist[i2][0] = 0
            if i > 0 and j > 0 and correct[i - 1] == non_correct[j - 1]:
                edit_dist[i2][j] = edit_dist[i2 ^ 1][j - 1]
                operation[i][j] = TYPO_NOCHANGE
            if i > 0 and edit_dist[i2][j] > edit_dist[i2 ^ 1][j] + 1:
                edit_dist[i2][j] = edit_dist[i2 ^ 1][j] + 1
                operation[i][j] = TYPO_DEL
            if j > 0 and edit_dist[i2][j] > edit_dist[i2][j - 1] + 1:
                edit_dist[i2][j] = edit_dist[i2][j - 1] + 1
                operation[i][j] = TYPO_ADD

            if edit_dist[i2][j] == edit_dist[i2][j - 1] + 1 == edit_dist[i2 ^ 1][j] + 1:
                if correct[i - 1] == ' ':
                    operation[i][j] = TYPO_DEL
                else:
                    operation[i][j] = TYPO_ADD

    state = (len(correct), len(non_correct))
    operations_sequence = []
    while state != (0, 0):
        i, j = state
        if operation[i][j] == TYPO_NOCHANGE:
            state = (i - 1, j - 1)
        elif operation[i][j] == TYPO_ADD:
            state = (i, j - 1)
            operations_sequence.append((i, TYPO_ADD, non_correct[j - 1]))
        elif operation[i][j] == TYPO_DEL:
            state = (i - 1, j)
            operations_sequence.append((i - 1, TYPO_DEL))
        else:
            assert False, "There should be a valid operation"
    operations_sequence = operations_sequence[::-1]
    reconstructed_non_correct = apply_operations(correct, operations_sequence)
    assert non_correct == reconstructed_non_correct, (
        non_correct, reconstructed_non_correct)
    return operations_sequence


def detailed_edit_operations_wrapper(tup):
    return detailed_edit_operations(*tup)


def detailed_edit_operations(correct, non_correct,
                             allow_change_operations=False):
    """
    Get detailed edit operations from the correct text to non-correct text,
    which are the edit alignments.

    :param str correct: Correct text
    :param str non_correct: Non-correct text
    :param bool allow_change_operations: Decide if CHG operations are allowed
    :rtype: list
    :returns: Sequence of detailed edit alignments operations
    """
    INF = len(correct) + len(non_correct) + 1
    edit_dist = [[INF for j in range(len(non_correct) + 1)]
                 for i in range(2)]
    if len(correct) * len(non_correct) > 5 * 10**6:
        operation = np.zeros((len(correct) + 1, len(non_correct) + 1),
                             dtype=np.int8)
    else:
        operation = [[None for j in range(len(non_correct) + 1)]
                     for i in range(len(correct) + 1)]
    # edit_dist = np.zeros((len(correct) + 1, len(non_correct) + 1),
    #                      dtype=np.int32)
    # edit_dist[:,:] = INF
    for i in range(len(correct) + 1):
        ii = i & 1
        edit_dist[ii] = [INF for j in range(len(non_correct) + 1)]
        for j in range(len(non_correct) + 1):
            if i == 0 and j == 0:
                edit_dist[0][0] = 0
            if i > 0 and j > 0 and correct[i - 1] == non_correct[j - 1]:
                edit_dist[ii][j] = edit_dist[ii ^ 1][j - 1]
                operation[i][j] = TYPO_NOCHANGE
            if j > 0 and edit_dist[ii][j] > edit_dist[ii][j - 1] + 1:
                edit_dist[ii][j] = edit_dist[ii][j - 1] + 1
                operation[i][j] = TYPO_ADD
            if (allow_change_operations and i > 0 and j > 0 and
                    correct[i - 1] != non_correct[j - 1] and
                    edit_dist[ii][j] > edit_dist[ii ^ 1][j - 1] + 1):
                edit_dist[ii][j] = edit_dist[ii ^ 1][j - 1] + 1
                operation[i][j] = TYPO_CHANGE
            if i > 0 and edit_dist[ii][j] > edit_dist[ii ^ 1][j] + 1:
                edit_dist[ii][j] = edit_dist[ii ^ 1][j] + 1
                operation[i][j] = TYPO_DEL

            if edit_dist[ii ^ 1][j] + 1 == edit_dist[ii][j - 1] + 1 == edit_dist[ii][j]:
                if correct[i - 1] == ' ':
                    operation[i][j] = TYPO_DEL
                else:
                    operation[i][j] = TYPO_ADD
    del edit_dist
    state = (len(correct), len(non_correct))
    operations_sequence = []
    while state != (0, 0):
        i, j = state
        if operation[i][j] == TYPO_NOCHANGE:
            state = (i - 1, j - 1)
            operations_sequence.append((i - 1, j - 1, TYPO_NOCHANGE))
        elif operation[i][j] == TYPO_ADD:
            state = (i, j - 1)
            operations_sequence.append((i, j - 1,
                                        (TYPO_ADD, non_correct[j - 1])))
        elif operation[i][j] == TYPO_DEL:
            state = (i - 1, j)
            operations_sequence.append((i - 1, j, TYPO_DEL))
        elif operation[i][j] == TYPO_CHANGE:
            state = (i - 1, j - 1)
            operations_sequence.append((i - 1, j - 1,
                                        (TYPO_CHANGE, non_correct[j - 1])))
        else:
            assert False, "There should be a valid operation"
    del operation
    operations_sequence = operations_sequence[::-1]
    return operations_sequence
