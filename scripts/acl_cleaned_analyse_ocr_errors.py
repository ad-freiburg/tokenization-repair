import sys
from typing import List, Tuple

import project
from src.helper.files import read_lines


from scripts.acl_cleaned_get_ocr_errors import longest_common_subsequence, unite_matched_spans, get_span_gaps


def get_ocr_character_edits(raw: str, cleaned: str) -> List[Tuple[str, str]]:
    matched = longest_common_subsequence(raw, cleaned)
    matched_spans = unite_matched_spans(matched)
    gaps = get_span_gaps(matched_spans, len(raw), len(cleaned))
    char_edits = []
    for gap_raw, gap_clean in gaps:
        err_raw = raw[gap_raw[0]:(gap_raw[1] + 1)]
        err_clean = cleaned[gap_clean[0]:(gap_clean[1] + 1)]
        char_edits.append((err_raw, err_clean))
    return char_edits


if __name__ == "__main__":
    #in_file = "/home/hertel/tokenization-repair-dumps/nastase/ocr_errors.txt"
    in_file = sys.argv[1]  # "icdar_ocr_errors.txt"
    print_readable = False

    error_frequencies = {}

    for l_i, line in enumerate(read_lines(in_file)):
        if "\t" in line:
            raw, cleaned = line.split("\t")
            char_edits = get_ocr_character_edits(raw, cleaned)
            for err_raw, err_clean in char_edits:
                #print(f"'{err_raw}' -> '{err_clean}'", gap_raw, gap_clean)
                if (err_raw, err_clean) not in error_frequencies:
                    error_frequencies[(err_raw, err_clean)] = 1
                else:
                    error_frequencies[(err_raw, err_clean)] += 1
        else:
            pass
            #print(line)

    errors = sorted(error_frequencies, key=lambda x: error_frequencies[x], reverse=True)
    for error in errors:
        if print_readable:
            print(f"'{error[0]}' -> '{error[1]}' {error_frequencies[error]}")
        else:
            print("\t".join([error[0], error[1], str(error_frequencies[error])]))
