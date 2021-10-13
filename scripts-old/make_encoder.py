import sys

from project import src
from src.helper.files import read_sequences
from src.helper.data_structures import select_most_frequent
from src.settings import symbols
from src.helper.pickle import dump_object


if __name__ == "__main__":
    text_file = sys.argv[1]
    out_file = sys.argv[2]

    char_frequencies = {}

    for i, line in enumerate(read_sequences(text_file)):
        for char in line:
            if char not in char_frequencies:
                char_frequencies[char] = 1
            else:
                char_frequencies[char] += 1
        if (i + 1) % 100000 == 0:
            print(i + 1, "lines", len(char_frequencies), "unique characters")
        if (i + 1) == 10000000:
            break

    chars = select_most_frequent(char_frequencies, 200)
    symbs = [symbols.SOS, symbols.EOS, symbols.UNKNOWN]
    encoder = {symbol: i for i, symbol in enumerate(sorted(chars) + symbs)}

    dump_object(encoder, out_file)
