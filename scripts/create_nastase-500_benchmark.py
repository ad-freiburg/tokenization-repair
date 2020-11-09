import sys

from project import src
from src.helper.files import read_lines


if __name__ == "__main__":
    mode = sys.argv[1]

    base_dir = "/home/hertel/tokenization-repair-dumps/data/benchmarks/nastase-500/"

    correct_sequences = read_lines(base_dir + "correct.txt")
    corrupt_sequences = read_lines(base_dir + "corrupt.txt")
    nastase_sequences = read_lines(base_dir + "nastase_shuffled.txt")

    for correct, corrupt, nastase in zip(correct_sequences, corrupt_sequences, nastase_sequences):
        if correct.startswith("==="):
            break
        elif correct.startswith("####"):
            continue
        elif correct.replace(" ", "") == corrupt.replace(" ", ""):
            if mode == "correct":
                print(correct)
            elif mode == "nastase":
                print(nastase)
            else:
                print(corrupt)
