from typing import Optional, Tuple

import sys
import random
import os

from project import src
from src.helper.files import read_lines

from create_pdfextract_benchmark import get_input_sequence


def standardise_spaces(text: str) -> str:
    return " ".join(text.split()).strip()


def remove_changeable_characters(text: str) -> str:
    return text.replace(" ", "").replace("-", "")


def match_line(true_line, input_lines) -> Optional[Tuple[int, int]]:
    for i, line in enumerate(input_lines):
        if true_line.startswith(line):
            j = i + 1
            matched = line
            while j < len(input_lines) and len(matched) < len(true_line):
                matched += input_lines[j]
                j += 1
            if true_line == matched:
                return (i, j)
    return None


if __name__ == "__main__":
    random.seed(5112020)

    input_files_directory = sys.argv[1]
    groundtruth_files_directory = sys.argv[2]
    files_file = sys.argv[3]
    out_path = sys.argv[4]
    groundtruth_file_suffix = ".body.txt"
    input_file_suffix = ".txt"  # ".final.txt"

    files = read_lines(files_file)

    n_matched = 0
    n_corrupt = 0

    sequence_pairs = []

    for file in files:
        true_path = groundtruth_files_directory + file
        input_path = input_files_directory + file[:-len(groundtruth_file_suffix)] + input_file_suffix
        if not os.path.exists(input_path):
            print("skipped %s (file not found)" % input_path)
            continue
        elif not os.path.exists(true_path):
            print("skipped %s (file not found)" % true_path)
            continue

        true_lines = read_lines(true_path)
        true_lines = [standardise_spaces(line) for line in true_lines]
        true_lines = [line for line in true_lines if len(line) > 0]

        input_lines = read_lines(input_path)
        input_lines = [standardise_spaces(line) for line in input_lines]
        input_lines = [line for line in input_lines if len(line) > 0]
        input_lines_removed = [remove_changeable_characters(line) for line in input_lines]

        print(file, len(true_lines), len(input_lines), n_matched)

        for line in true_lines:
            line_removed = remove_changeable_characters(line)
            matched = match_line(line_removed, input_lines_removed)
            if matched is not None:
                begin, end = matched
                matched = " ".join(input_lines[begin:end])
                input_sequence = get_input_sequence(line, matched)
                if input_sequence is not None and input_sequence.replace(" ", "") == line.replace(" ", ""):
                    n_matched += 1
                    if input_sequence != line:
                        #print(line)
                        #print(input_sequence)
                        #print()
                        n_corrupt += 1
                    sequence_pairs.append((input_sequence, line))
                else:
                    pass
                    """print(line)
                    print(matched)
                    print(input_sequence)
                    print()"""

    print(n_matched, "matched paragraphs")
    print("%i corrupt sequences (%.2f%%)" % (n_corrupt, n_corrupt / n_matched * 100))

    random.shuffle(sequence_pairs)

    with open(out_path + "correct_all.txt", "w") as correct_file, open(out_path + "corrupt_all.txt", "w") as corrupt_file:
        for corrupt, correct in sequence_pairs:
            correct_file.write(correct + "\n")
            corrupt_file.write(corrupt + "\n")
