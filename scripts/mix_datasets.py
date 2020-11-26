import sys
import random


ENCODING = "utf8"
ERRORS = "replace"


def num_lines(file):
    n = 0
    with open(file, encoding=ENCODING, errors=ERRORS) as f:
        for _ in f:
            n += 1
    return n


if __name__ == "__main__":
    random.seed(19112020)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    out_file = sys.argv[3]

    n1 = num_lines(file1)
    print(n1, "lines")
    n2 = num_lines(file2)
    print(n2, "lines")

    print("Shuffling indices...")
    file_indices = [True] * n1 + [False] * n2
    random.shuffle(file_indices)

    file1 = open(file1, encoding=ENCODING, errors=ERRORS)
    file2 = open(file2, encoding=ENCODING, errors=ERRORS)
    out_file = open(out_file, "w", encoding=ENCODING, errors=ERRORS)

    print("Writing...")
    for ii, i in enumerate(file_indices):
        if i:
            line = next(file1)
        else:
            line = next(file2)
        out_file.write(line)
        if (ii + 1) % 100000 == 0:
            print(ii + 1, "lines")

    for file in (file1, file2, out_file):
        file.close()
