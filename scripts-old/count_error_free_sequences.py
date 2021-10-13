import sys

import project
from src.helper.files import read_lines


if __name__ == "__main__":
    lines_a = read_lines(sys.argv[1])
    lines_b = read_lines(sys.argv[2])
    n = len(lines_a)
    n_equal = 0
    for a, b in zip(lines_a, lines_b):
        if a.replace("-", "") == b.replace("-", ""):
            n_equal += 1
    print(n_equal / n)
    print(n_equal)
    print(n)
