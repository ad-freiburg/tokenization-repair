import sys
import random

import project
from src.helper.files import read_sequences
from src.helper.stochastic import flip_coin


def hyphenate(token):
    if len(token) < 2:
        return token
    positions = []
    for i in range(0, len(token) - 1):
        if token[i:(i + 2)].isalpha():
            positions.append(i)
    if len(positions) > 0:
        hyphen_pos = random.choice(positions) + 1
        token = token[:hyphen_pos] + "-" + token[hyphen_pos:]
    return token


def introduce_hyphens(sequence, p):
    tokens = sequence.split(" ")
    for i in range(len(tokens)):
        if flip_coin(random, p):
            tokens[i] = hyphenate(tokens[i])
    sequence = " ".join(tokens)
    return sequence


if __name__ == "__main__":
    hyphenation_rate = 0.0114

    in_file = sys.argv[1]
    out_file = sys.argv[2]

    out_file = open(out_file, "w")

    for i, line in enumerate(read_sequences(in_file)):
        line = introduce_hyphens(line, hyphenation_rate)
        out_file.write(line + "\n")
        if i % 10000 == 0:
            print(i, "lines")

    out_file.close()
