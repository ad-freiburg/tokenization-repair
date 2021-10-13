import sys

from project import src
from src.helper.pickle import load_object
from src.settings import paths
from src.helper.data_structures import sort_dict_by_value
from src.interactive.sequence_generator import interactive_sequence_generator


if __name__ == "__main__":
    frequencies = load_object(paths.CHARACTER_FREQUENCY_DICT)
    sorted_frequencies = sort_dict_by_value(frequencies)
    ranks = {char: i for i, (char, frequency) in enumerate(sorted_frequencies)}

    if len(sys.argv) > 1:
        print_top_n = int(sys.argv[1])
        for i, (frequency, char) in enumerate(sorted_frequencies[:print_top_n]):
            print(i, char, frequency)
    else:
        for char in interactive_sequence_generator():
            freq = frequencies[char]
            rank = ranks[char]
            print("rank %i (frequency %i)" % (rank, freq))
