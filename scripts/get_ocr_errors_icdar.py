import project
from src.helper.files import read_lines


if __name__ == "__main__":
    dir = "~/tokenization-repair-dumps/data/spelling/icdar2017.periodical/"
    correct_lines = read_lines(dir + "spelling.txt")
    corrupt_lines = read_lines(dir + "corrupt.txt")

    for correct, corrupt in zip(correct_lines, corrupt_lines):
        print(correct)
        print(corrupt)