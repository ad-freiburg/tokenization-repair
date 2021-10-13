import project
from src.datasets.wikipedia import Wikipedia
from src.settings import paths
from src.helper.pickle import dump_object


K = 100000
M = 10 * K


if __name__ == "__main__":
    token_frequencies = {}
    for s_i, sequence in enumerate(Wikipedia.training_sequences()):
        if s_i % K == 0:
            print("%.1fM sequences, %.1fM tokens" % (s_i / M, len(token_frequencies) / M))
        tokens = sequence.split()
        for token in tokens:
            if token not in token_frequencies:
                token_frequencies[token] = 1
            else:
                token_frequencies[token] += 1
    dump_object(token_frequencies, paths.TOKEN_FREQUENCY_DICT)
