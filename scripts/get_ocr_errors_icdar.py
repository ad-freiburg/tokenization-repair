import sys

import project
from src.helper.files import read_lines


from acl_cleaned_get_ocr_errors import get_ocr_errors


if __name__ == "__main__":
    dir = sys.argv[1]  # "/home/hertel/tokenization-repair-dumps/data/spelling/icdar2017.periodical/development/"
    correct_lines = read_lines(dir + "spelling.txt")
    corrupt_lines = read_lines(dir + "corrupt.txt")

    for correct, corrupt in zip(correct_lines, corrupt_lines):
        #print(correct)
        #print(corrupt)
        raw_tokens = corrupt.split()
        clean_tokens = correct.split()
        ocr_errors = get_ocr_errors(raw_tokens, clean_tokens)
        for wrong, right in ocr_errors:
            print(wrong + "\t" + right)