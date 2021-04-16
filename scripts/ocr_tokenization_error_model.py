import sys
import random
from enum import Enum

import project
from src.helper.files import read_lines


class SequenceType(Enum):
    NO_ERRORS = 0
    FEW_ERRORS = 1
    MANY_ERRORS = 2


SEQUENCE_TYPES = list(SequenceType)


def sample_sequence_type() -> SequenceType:
    return random.choice(SEQUENCE_TYPES)


def sample_prob(sequence_type: SequenceType):
    if sequence_type == SequenceType.NO_ERRORS:
        return 0
    if sequence_type == SequenceType.FEW_ERRORS:
        return random.uniform(0.01, 0.1)
    return random.uniform(0.1, 1)


def toss_coin(p: float) -> bool:
    return random.random() < p


def corrupt_tokenization(sequence: str) -> str:
    sequence_type = sample_sequence_type()
    if sequence_type == SequenceType.NO_ERRORS:
        return sequence
    p = sample_prob(sequence_type)
    corrupt = ""
    for i, char in enumerate(sequence):
        if char == " ":
            if not toss_coin(p):
                corrupt += char
        else:
            if i > 0 and sequence[i - 1] != " " and toss_coin(p):
                corrupt += " "
            corrupt += char
    return corrupt


if __name__ == "__main__":
    random.seed(42)

    in_file = sys.argv[1]

    for line in read_lines(in_file):
        corrupt = corrupt_tokenization(line)
        print(corrupt)
