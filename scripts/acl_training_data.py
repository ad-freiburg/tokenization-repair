import sys
import random

import project
from src.helper.files import get_files, read_lines
from select_acl_articles import get_year


def preprocess(line: str):
    tokens = [token for token in line.split() if len(token) > 0]
    line = " ".join(tokens)
    return line


if __name__ == "__main__":
    random.seed(42)

    acl_dir = "/home/hertel/tokenization-repair-dumps/nastase/acl-201302_word-resegmented/raw/"

    files = sorted(get_files(acl_dir))

    files = [file for file in files if get_year(file) >= 2005]
    examples = []

    for file in files:
        lines = read_lines(acl_dir + file)
        lines = [preprocess(line) for line in lines]
        lines = [line for line in lines if len(line) > 0]
        examples.extend(lines)

    random.shuffle(examples)
    for line in examples:
        print(line)
