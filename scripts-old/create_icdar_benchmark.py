import os
import sys

import project
from src.helper.files import read_lines


def create_input(corrupt, aligned, truth):
    input_sequences = []
    ground_truth_sequences = []
    input_sequence = ""
    ground_truth_sequence = ""
    i = 0
    j = 0
    while i < len(corrupt):
        if aligned[j] == ALIGNMENT_SYMBOL:
            ground_truth_sequence += truth[j]
            j += 1
        elif truth[j] == ALIGNMENT_SYMBOL:
            input_sequence += corrupt[i]
            i += 1
            j += 1
        elif truth[j] == NO_GS_SYMBOL:
            i += 1
            j += 1
            input_sequence = input_sequence.strip()
            ground_truth_sequence = ground_truth_sequence.strip()
            if len(input_sequence) > 0 or len(ground_truth_sequence) > 0:
                input_sequences.append(input_sequence)
                ground_truth_sequences.append(ground_truth_sequence)
                input_sequence = ""
                ground_truth_sequence = ""
        else:
            input_sequence += corrupt[i]
            ground_truth_sequence += truth[j]
            i += 1
            j += 1
    input_sequence = input_sequence.strip()
    ground_truth_sequence = ground_truth_sequence.strip()
    if len(input_sequence) > 0 or len(ground_truth_sequence) > 0:
        input_sequences.append(input_sequence)
        ground_truth_sequences.append(ground_truth_sequence)
    return input_sequences, ground_truth_sequences


if __name__ == "__main__":
    directory = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None

    files = os.listdir(directory)

    prefix_input =   "[OCR_toInput] "
    prefix_aligned = "[OCR_aligned] "
    prefix_truth =   "[ GS_aligned] "

    NO_GS_SYMBOL = "#"
    ALIGNMENT_SYMBOL = "@"

    if out_dir is not None:
        corrupt_file = open(out_dir + "/corrupt.txt", "w")
        ground_truth_file = open(out_dir + "/spelling.txt", "w")

    for file in files:
        print(file)
        corrupt, aligned, truth = read_lines(directory + "/" + file)
        corrupt = corrupt[len(prefix_input):]
        aligned = aligned[len(prefix_aligned):]
        truth = truth[len(prefix_truth):]
        print(len(corrupt), len(aligned), len(truth))
        input_sequences, ground_truth_sequences = create_input(corrupt, aligned, truth)
        for s_in, s_true in zip(input_sequences, ground_truth_sequences):
            print(s_in)
            print(s_true)
            if out_dir is not None:
                corrupt_file.write(s_in + "\n")
                ground_truth_file.write(s_true + "\n")

    if out_dir is not None:
        corrupt_file.close()
        ground_truth_file.close()