from typing import Tuple, List

import numpy as np
from enum import Enum

from src.spelling.evaluation import TokenErrorType, longest_common_subsequence


class EditOperation(Enum):
    KEEP = 0
    INSERT = 1
    DELETE = 2
    REPLACE = 3


def edit_distance(s: str, t: str, substitutions=False) -> Tuple[int, List[Tuple[int, EditOperation, str]]]:
    n = len(s)
    m = len(t)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[0, :] = range(m + 1)
    d[:, 0] = range(n + 1)
    nospace_ops = np.zeros_like(d)
    edit_operations = [[None] + [EditOperation.INSERT] * m]
    for i, s_char in enumerate(s):
        edit_operations.append([EditOperation.DELETE])
        for j, t_char in enumerate(t):
            # insert
            cost = d[i + 1, j] + 1, nospace_ops[i + 1, j] + (1 if t_char != " " else 0)
            action = EditOperation.INSERT
            # delete
            delete_cost = d[i, j + 1] + 1, nospace_ops[i, j + 1] + (1 if s_char != " " else 0)
            if delete_cost < cost:
                cost = delete_cost
                action = EditOperation.DELETE
            # keep
            if s_char == t_char:
                keep_cost = d[i, j], nospace_ops[i, j]
                if keep_cost <= cost:
                    cost = keep_cost
                    action = EditOperation.KEEP
            # replace
            if substitutions:
                if s_char != " " and t_char != " ":
                    replace_cost = d[i, j] + 1, nospace_ops[i, j] + 1
                    if replace_cost <= cost:
                        cost = replace_cost
                        action = EditOperation.REPLACE
            d[i + 1, j + 1] = cost[0]
            nospace_ops[i + 1, j + 1] = cost[1]
            edit_operations[-1].append(action)
    """print(d)
    print(nospace_ops)
    for ops in edit_operations:
        print(ops)"""
    # backtrace
    backtrace = []
    i = n
    j = m
    while i > 0 or j > 0:
        operation = edit_operations[i][j]
        if operation == EditOperation.INSERT:
            char = t[j - 1]
            j = j - 1
        elif operation == EditOperation.DELETE:
            char = s[i - 1]
            i = i - 1
        else:
            char = (t[j - 1], s[i - 1])
            i = i - 1
            j = j - 1
        if operation != EditOperation.KEEP:
            backtrace.append((i, operation, char))
    return d[-1, -1], backtrace[::-1]


def get_labels(sequence: str, edit_operations: List[Tuple[int, EditOperation, str]]) -> List[TokenErrorType]:
    space_edit_positions = {pos for pos, op, char in edit_operations if char == " "}
    nospace_insertion_positions = {pos for pos, op, char in edit_operations if char != " " and op == EditOperation.INSERT}
    nospace_edit_positions = {pos for pos, op, char in edit_operations if char != " " and op != EditOperation.INSERT}
    tokens = sequence.split(" ")
    labels = []
    pos = 0
    for token in tokens:
        end = pos + len(token)
        space_edited = False
        for i in range(pos - 1, end + 1):
            if i in space_edit_positions:
                space_edited = True
                break
        nospace_edited = False
        for i in range(pos + 1, end):
            if i in nospace_edit_positions or i in nospace_insertion_positions:
                nospace_edited = True
                break
        if pos in nospace_edit_positions or end in nospace_edit_positions:
            nospace_edited = True
        if pos in nospace_insertion_positions or end in nospace_insertion_positions:
            nospace_edited = True
        pos = end + 1
        if space_edited and nospace_edited:
            label = TokenErrorType.MIXED
        elif space_edited:
            label = TokenErrorType.TOKENIZATION_ERROR
        elif nospace_edited:
            label = TokenErrorType.OCR_ERROR
        else:
            label = TokenErrorType.NONE
        labels.append(label)
    return labels


def get_token_edit_labels(correct: str, corrupt: str) -> List[TokenErrorType]:
    correct_tokens = correct.split(" ")
    if correct == corrupt:
        return [TokenErrorType.NONE] * len(correct_tokens)
    d, operations = edit_distance(correct, corrupt)
    labels = get_labels(correct, operations)
    corrupt_tokens = corrupt.split(" ")
    matched = longest_common_subsequence(correct_tokens, corrupt_tokens)
    for i, j in matched:
        labels[i] = TokenErrorType.NONE
    return labels