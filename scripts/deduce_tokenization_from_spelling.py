import sys

import project
from src.settings import paths
from src.helper.files import read_lines
from spelling_evaluation_space_preference import edit_distance, EditOperation


def deduce_tokenization(corrupt, predicted):
    if corrupt == predicted:
        return corrupt
    d, edits = edit_distance(corrupt, predicted, substitutions=True)
    insertion_positions = set()
    deletion_positions = set()
    for edit in edits:
        pos, type, char = edit
        if char == " ":
            if type == EditOperation.INSERT:
                insertion_positions.add(pos)
            else:
                deletion_positions.add(pos)
    deduced = ""
    for i, char in enumerate(corrupt):
        if i in insertion_positions:
            deduced += " "
        if i not in deletion_positions:
            deduced += char
    return deduced


if __name__ == "__main__":
    benchmark = sys.argv[1]
    subset = sys.argv[2]

    input_file = paths.BENCHMARKS_DIR + benchmark + "/" + subset + "/corrupt.txt"
    predicted_file = paths.DUMP_DIR + "spelling/" + benchmark + "/" + subset + "/google.txt"
    out_file = paths.RESULTS_DIR + benchmark + "/" + subset + "/google_deduced.txt"

    input_sequences = read_lines(input_file)
    predicted_sequences = read_lines(predicted_file)

    with open(out_file, "w") as f:
        for corrupt, predicted in zip(input_sequences, predicted_sequences):
            tokenized = deduce_tokenization(corrupt, predicted)
            if corrupt != predicted:
                print(corrupt)
                print(predicted)
                print(tokenized)
            f.write(tokenized + "\n")
