import sys

import project
from src.helper.files import read_lines


def remove_hyphenation(sequence):
    i = 0
    removed = ""
    while i < len(sequence):
        if i > 0 and i + 2 < len(sequence) and sequence[i - 1].isalpha() and sequence[i:(i + 2)] in ("- ", "‑ ") \
                and sequence[i + 2].isalpha():
            i += 2
        else:
            removed += sequence[i]
            i += 1
    return removed


def correct_ground_truth(sequence):
    sequence = sequence.replace("ſ", "s")
    sequence = sequence.replace("ﬁ", "fi")
    sequence = sequence.replace("ﬀ", "ff")
    sequence = sequence.replace("ﬃ", "ffi")
    for punctuation in ",;?!:)”":
        sequence = sequence.replace(" " + punctuation, punctuation)
    for punctuation in "(“":
        sequence = sequence.replace(punctuation + " ", punctuation)
    sequence = remove_hyphenation(sequence)
    return sequence


if __name__ == "__main__":
    in_file = sys.argv[1]

    for line in read_lines(in_file):
        correct = correct_ground_truth(line)
        print(correct)
