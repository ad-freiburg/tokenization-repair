from src.helper.pickle import load_object
from src.settings import paths
from src.helper.data_structures import select_most_frequent
from src.helper.pickle import load_object, dump_object
from src.settings import paths
from src.helper.files import file_exists


def load_most_frequent(n):
    path = None
    if n is not None:
        path = paths.MOST_FREQUENT_UNIGRAMS_DICT % n
        if file_exists(path):
            frequencies = load_object(path)
            return frequencies
    delim_frequencies = load_object(paths.UNIGRAM_DELIM_FREQUENCY_DICT)
    no_delim_frequencies = load_object(paths.UNIGRAM_NO_DELIM_FREQUENCY_DICT)
    frequencies = delim_frequencies
    for token in no_delim_frequencies:
        if token not in frequencies:
            frequencies[token] = no_delim_frequencies[token]
        else:
            frequencies[token] += no_delim_frequencies[token]
    if n is not None:
        frequencies = select_most_frequent(frequencies, n)
        dump_object(frequencies, path)
    return frequencies


class UnigramHolder:
    def __init__(self, n=None):
        self.frequencies = load_most_frequent(n)

    def __len__(self):
        return len(self.frequencies)

    def is_unigram(self, text: str):
        return text in self.frequencies

    def get(self, text: str):
        if not self.is_unigram(text):
            return 0
        return self.frequencies[text]

