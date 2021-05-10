import sys
import matplotlib.pyplot as plt
import math
from hyphen import Hyphenator

import project
from src.helper.files import read_lines

from acl_cleaned_get_ocr_errors import get_ocr_errors
from acl_cleaned_analyse_ocr_errors import get_ocr_character_edits


if __name__ == "__main__":
    raw_file = sys.argv[1]
    clean_file = sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else None

    hyphenator = Hyphenator()

    error_frequencies = {}

    for i, (corrupt, correct) in enumerate(zip(read_lines(raw_file), read_lines(clean_file))):
        print(f"** SEQUENCE {i} **")
        corrupt_tokens = corrupt.split()
        correct_tokens = correct.split()
        ocr_errors = get_ocr_errors(corrupt_tokens, correct_tokens)
        for corrupt, correct in ocr_errors:
            corrupt_parts = corrupt.split(" ")
            correct_parts = correct.split(" ")
            if len(corrupt_parts) != len(correct_parts):
                continue
            for corrupt_part, correct_part in zip(corrupt_parts, correct_parts):
                edits = get_ocr_character_edits(correct_part, corrupt_part)
                edits = [e for e in edits if e != ("", "-")]
                if len(edits) > 0:
                    n_char_edits = []
                    token_len = len(correct_part)
                    n_errors = len(edits)
                    if token_len not in error_frequencies:
                        error_frequencies[token_len] = {}
                    if n_errors not in error_frequencies[token_len]:
                        error_frequencies[token_len][n_errors] = 1
                    else:
                        error_frequencies[token_len][n_errors] += 1
                    print(correct_part, corrupt_part, edits, "len=%i" % token_len, "errors=%i" % n_errors)

    for token_len in sorted(error_frequencies):
        print("len = %i" % token_len)
        for n_errors in sorted(error_frequencies[token_len]):
            print(f"  {n_errors} errors = {error_frequencies[token_len][n_errors]}")
