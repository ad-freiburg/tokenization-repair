import project
from src.helper.files import read_lines
from src.spelling.evaluation import longest_common_subsequence
from acl_cleaned_get_ocr_errors import get_span_gaps, unite_matched_spans, get_ocr_errors
from acl_cleaned_analyse_ocr_errors import get_ocr_character_edits


def get_erroneous_spans(corrupt_sequence, correct_sequence):
    corrupt_tokens = corrupt_sequence.split(" ")
    correct_tokens = correct_sequence.split(" ")
    matching = longest_common_subsequence(corrupt_tokens, correct_tokens)
    matched_spans = unite_matched_spans(matching)
    span_gaps = get_span_gaps(matched_spans, len(corrupt_tokens), len(correct_tokens))
    span_pairs = []
    for corrupt_span, correct_span in span_gaps:
        corrupt_text = " ".join(corrupt_tokens[corrupt_span[0]:(corrupt_span[1] + 1)])
        correct_text = " ".join(correct_tokens[correct_span[0]:(correct_span[1] + 1)])
        span_pairs.append((corrupt_text, correct_text))
    return span_pairs


def is_space_edit(pair):
    return " " in pair[0] or " " in pair[1]


def is_ocr_edit(pair):
    return pair[0].replace(" ", "") != "" or pair[1].replace(" ", "") != ""


if __name__ == "__main__":
    path = "benchmarks/ACL/development/"
    corrupt_file = path + "corrupt.txt"
    ground_truth_file = path + "spelling.txt"

    out_path = "char_error_distributions/"
    out_file_ocr = out_path + "span_ocr_error_rates.txt"
    out_file_tokenization = out_path + "span_tokenization_error_rates.txt"
    out_file_spans = out_path + "spans.txt"
    out_file_ocr = open(out_file_ocr, "w")
    out_file_tokenization = open(out_file_tokenization, "w")
    out_file_spans = open(out_file_spans, "w")

    for i, (corrupt, correct) in enumerate(zip(read_lines(corrupt_file), read_lines(ground_truth_file))):
        #print(i)
        #print(corrupt)
        #print(correct)
        erroneous_spans = get_erroneous_spans(corrupt, correct)
        for corrupt_span, correct_span in erroneous_spans:
            if corrupt_span.replace("-", "") != correct_span and len(correct_span) > 0:
                n_tokens = correct_span.count(" ") + 1
                char_edits = get_ocr_character_edits(corrupt_span, correct_span)
                n_chars = len(correct_span)
                n_space_edits = 0
                n_space_insertions = 0
                n_space_deletions = 0
                n_ocr_edits = 0
                for edit in char_edits:
                    if is_space_edit(edit):
                        n_space_edits += 1
                        if " " in edit[0]:
                            n_space_insertions += 1
                        if " " in edit[1]:
                            n_space_deletions += 1
                    if is_ocr_edit(edit):
                        n_ocr_edits += 1
                possible_tokenization_errors = n_chars - n_tokens
                n_nospace_chars = n_chars - n_tokens + 1
                tokenization_error_rate = 0 if possible_tokenization_errors == 0 else n_space_edits / possible_tokenization_errors
                ocr_error_rate = n_ocr_edits / n_nospace_chars
                print("\t".join([str(val) for val in
                                 [correct_span, corrupt_span, n_tokens, tokenization_error_rate, ocr_error_rate]]))
                print("\t", char_edits)
                if tokenization_error_rate > 0:
                    out_file_tokenization.write(str(tokenization_error_rate) + "\n")
                if ocr_error_rate > 0:
                    out_file_ocr.write(str(ocr_error_rate) + "\n")
                n_spaces = n_tokens - 1
                possible_space_insertions = n_chars - 2 * n_tokens + 1
                space_insertion_rate = n_space_insertions / possible_space_insertions \
                    if possible_space_insertions > 0 else 0
                space_deletion_rate = n_space_deletions / (n_tokens - 1) if n_tokens > 1 else 0
                out_file_spans.write("\t".join([str(val) for val in
                                                [n_tokens, space_insertion_rate, space_deletion_rate, ocr_error_rate]]))
                out_file_spans.write("\n")

    out_file_ocr.close()
    out_file_tokenization.close()
