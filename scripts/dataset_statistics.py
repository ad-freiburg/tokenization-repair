"""
Prints statistics about the Wikipedia dataset.
"""

from itertools import chain
import re

from project import src
from src.data.wikipedia import Wikipedia


if __name__ == "__main__":
    sequences = chain(Wikipedia.training_sequences(),
                      Wikipedia.development_sequences(),
                      Wikipedia.test_sequences())

    n_sequences = 0
    n_tokens = 0
    n_characters = 0
    unique_characters = set()
    unique_words = set()

    for sequence in sequences:
        n_sequences += 1
        if n_sequences % 100000 == 0:
            print("%i sequences..." % n_sequences)
        n_tokens += len(sequence.split(' '))
        n_characters += len(sequence)
        sequence_words = re.findall(r"\w+", sequence)
        for word in sequence_words:
            unique_words.add(word)
        for char in sequence:
            unique_characters.add(char)

    print("%i sequences" % n_sequences)
    print("%i tokens" % n_tokens)
    print("%i characters" % n_characters)
    print("%.1f mean sequence length [tokens]" % (n_tokens / n_sequences))
    print("%.1f mean sequence length [characters]" % (n_characters / n_sequences))
    print("%i unique words" % len(unique_words))
    print("%i unique characters" % len(unique_characters))
