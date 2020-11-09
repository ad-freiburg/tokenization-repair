from typing import List, Tuple

from os import listdir
from src.settings import paths


TRUTH_FILE_SUFFIX = "body.txt"
INPUT_FILE_SUFFIX = "final.txt"


def get_files():
    files = []
    for folder in sorted(listdir(paths.ARXIV_GROUND_TRUTH_DIR)):
        for file in sorted(listdir(paths.ARXIV_GROUND_TRUTH_DIR + folder)):
            if file.endswith(TRUTH_FILE_SUFFIX):
                files.append(folder + "/" + file)
    return files


def read_lines(path: str) -> List[str]:
    with open(path) as f:
        lines = f.readlines()
    lines = [line[:-1] for line in lines]
    lines = [" ".join(line.split()) for line in lines]
    # TODO lines = [line for line in lines if len(line) > 0]
    return lines


def match_lines(truth_path: str, input_path: str) -> List[Tuple[str, str]]:
    true_lines = read_lines(truth_path)
    true_dict = {line.replace(" ", ""): line for line in true_lines}
    pairs = []
    for input_line in read_lines(input_path):
        input_no_spaces = input_line.replace(" ", "")
        if input_no_spaces in true_dict:
            pairs.append((input_line, true_dict[input_no_spaces]))
    return pairs


def to_input_file(true_file_name):
    return true_file_name[:-len(TRUTH_FILE_SUFFIX)] + INPUT_FILE_SUFFIX
