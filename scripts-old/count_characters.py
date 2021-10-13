"""
Counts the occurences of all characters in the Wikipedia training set and stores the counts on disk.
"""

import sys

from project import src
from src.data.wikipedia import Wikipedia
from src.helper.pickle import dump_object
from src.settings import paths


def count_chars(counters, sequence):
    for char in sequence:
        if char not in counters:
            counters[char] = 1
        else:
            counters[char] += 1


if __name__ == "__main__":
    n_sequences = int(sys.argv[1])
    char_counters = {}
    for i, sequence in enumerate(Wikipedia.training_sequences(n_sequences)):
        count_chars(char_counters, sequence)
        if (i + 1) % 100000 == 0:
            print("%i sequences processed" % (i + 1))
    print(char_counters)
    dump_object(char_counters, paths.CHARACTER_FREQUENCY_DICT)
