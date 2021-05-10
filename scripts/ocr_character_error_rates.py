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

    char_error_rates = []
    total_hyphen_edits = 0
    total_tokens = 0
    hyphenable_tokens = 0

    for i, (corrupt, correct) in enumerate(zip(read_lines(raw_file), read_lines(clean_file))):
        corrupt_tokens = corrupt.split()
        correct_tokens = correct.split()
        total_tokens += len(correct_tokens)
        for t in correct_tokens:
            try:
                if len(hyphenator.pairs(t)) > 0:
                    hyphenable_tokens += 1
            except IndexError:
                pass
        ocr_errors = get_ocr_errors(corrupt_tokens, correct_tokens)
        print(i + 1, ocr_errors)
        n_char_edits = 0
        n_hyphen_edits = 0
        for erroneous, corrected in ocr_errors:
            char_edits = get_ocr_character_edits(erroneous, corrected)
            char_edits = [edit for edit in char_edits if edit[0] != " " and edit[1] != " "]
            hyphen_edits = len([edit for edit in char_edits if edit == ("-", "")])
            print("", char_edits)
            n_char_edits += len(char_edits) - hyphen_edits
            n_hyphen_edits += hyphen_edits
        n_nonspace_chars = sum([len(token) for token in correct_tokens])
        error_rate = n_char_edits / n_nonspace_chars
        print(error_rate, n_char_edits, n_nonspace_chars, n_hyphen_edits)
        char_error_rates.append(error_rate)
        total_hyphen_edits += n_hyphen_edits

    print(char_error_rates)
    print(f"hyphenation rate: {total_hyphen_edits / hyphenable_tokens:.4f} ({total_hyphen_edits} / {hyphenable_tokens} "
          f"/ {total_tokens})")

    if out_file is not None:
        with open(out_file, "w") as f:
            for error_rate in char_error_rates:
                f.write(str(error_rate) + "\n")

    plt.hist(char_error_rates, bins=[-0.01, 1e-16] + [i / 100 for i in range(1, math.ceil(max(char_error_rates) * 100))])
    plt.show()
