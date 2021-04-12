import project
from src.helper.files import read_lines


from scripts.acl_cleaned_get_ocr_errors import longest_common_subsequence, unite_matched_spans, get_span_gaps


if __name__ == "__main__":
    in_file = "/home/hertel/tokenization-repair-dumps/nastase/ocr_errors.txt"
    print_readable = False

    error_frequencies = {}

    for l_i, line in enumerate(read_lines(in_file)):
        if "\t" in line:
            raw, cleaned = line.split("\t")
            matched = longest_common_subsequence(raw, cleaned)
            matched_spans = unite_matched_spans(matched)
            gaps = get_span_gaps(matched_spans, len(raw), len(cleaned))
            #print(line)
            for gap_raw, gap_clean in gaps:
                err_raw = raw[gap_raw[0]:(gap_raw[1] + 1)]
                err_clean = cleaned[gap_clean[0]:(gap_clean[1] + 1)]
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
