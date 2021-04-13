import sys

import project
from src.helper.files import read_lines


if __name__ == "__main__":
    lines = read_lines(sys.argv[1])
    for line in lines:
        print(line.replace(" ", ""))
