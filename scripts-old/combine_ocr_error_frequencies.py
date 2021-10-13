import sys

import project
from src.helper.files import read_lines


if __name__ == "__main__":
    n_files = len(sys.argv) // 2
    weighted_files = [(sys.argv[2 * i + 1], int(sys.argv[2 * (i + 1)])) for i in range(n_files)]

    ocr_error_frequencies = {}

    for file, weight in weighted_files:
        for line in read_lines(file):
            corrupt, correct, frequency = line.split("\t")
            frequency = int(frequency) * weight
            pair = (corrupt, correct)
            if pair in ocr_error_frequencies:
                ocr_error_frequencies[pair] += frequency
            else:
                ocr_error_frequencies[pair] = frequency

    pairs = sorted(ocr_error_frequencies, key=lambda pair: ocr_error_frequencies[pair], reverse=True)
    for pair in pairs:
        print("\t".join([pair[0], pair[1], str(ocr_error_frequencies[pair])]))
