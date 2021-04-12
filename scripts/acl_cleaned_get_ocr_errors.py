import math
from typing import List, Tuple

import os
import multiprocessing as mp
import argparse

import project
from src.spelling.evaluation import longest_common_subsequence


def read_tokens(file):
    with open(file, errors="ignore") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        new_line = line.endswith("\n")
        if new_line:
            line = line[:-1]
        tokens.extend(line.split())
        if new_line:
            tokens.append("\n")
    return tokens


def unite_matched_spans(matching: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    spans = []
    if len(matching) == 0:
        return spans
    span_start = matching[0]
    span_end = matching[0]
    for i, j in matching[1:]:
        if i == span_end[0] + 1 and j == span_end[1] + 1:
            span_end = i, j
        else:
            spans.append(((span_start[0], span_end[0]), (span_start[1], span_end[1])))
            span_start = i, j
            span_end = i, j
    spans.append(((span_start[0], span_end[0]), (span_start[1], span_end[1])))
    return spans


def get_span_gaps(matched_spans: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                  seq1_len: int,
                  seq2_len: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    gaps = []
    pos1 = 0
    pos2 = 0
    for span1, span2 in matched_spans:
        if span1[0] > 0 or span2[0] > 0:
            gap1 = (pos1, span1[0] - 1)
            gap2 = (pos2, span2[0] - 1)
            gaps.append((gap1, gap2))
        pos1 = span1[1] + 1
        pos2 = span2[1] + 1
    if pos1 < seq1_len or pos2 < seq2_len:
        gap1 = (pos1, seq1_len - 1)
        gap2 = (pos2, seq2_len - 1)
        gaps.append((gap1, gap2))
    return gaps


def preprocess(tokens: List[str]):
    return [t for t in tokens if t != "\n" and t != "\t"]


def get_ocr_errors(raw_dir: str, cleaned_dir: str, file: str) -> List[Tuple[str, str]]:
    raw_tokens = read_tokens(raw_dir + file)
    cleaned_tokens = read_tokens(cleaned_dir + file)
    matching_subsequence = longest_common_subsequence(raw_tokens, cleaned_tokens)
    matched_spans = unite_matched_spans(matching_subsequence)
    span_gaps = get_span_gaps(matched_spans, len(raw_tokens), len(cleaned_tokens))
    ocr_errors = []
    for raw_span, cleaned_span in span_gaps:
        unmatched_raw = preprocess(raw_tokens[raw_span[0]:(raw_span[1] + 1)])
        unmatched_cleaned = preprocess(cleaned_tokens[cleaned_span[0]:(cleaned_span[1] + 1)])
        if len(unmatched_raw) > 0 and len(unmatched_cleaned) > 0:
            char_len_ratio = len(''.join(unmatched_raw)) / len(''.join(unmatched_cleaned))
            if 0.5 <= char_len_ratio <= 2:
                ocr_errors.append((" ".join(unmatched_raw), " ".join(unmatched_cleaned)))
    return ocr_errors


def main(args):
    raw_dir = args.raw_directory
    cleaned_dir = args.clean_directory
    out_file = args.out_file

    files = sorted(os.listdir(cleaned_dir))
    files = [file for file in files if os.path.exists(raw_dir + file)]

    n_cpus = mp.cpu_count()
    batch_size = 4 * n_cpus
    batches = math.ceil(len(files) / batch_size)

    with open(out_file, "w") as out_file:
        for b_i in range(batches):
            start = b_i * batch_size
            end = start + batch_size
            arguments = [[raw_dir, cleaned_dir, files[i]] for i in range(start, end)]

            with mp.Pool(n_cpus) as pool:
                results = pool.starmap(get_ocr_errors, arguments)

            for k, result in enumerate(results):
                file = files[start + k]
                print(file)
                out_file.write(file + "\n")
                for raw, cleaned in result:
                    error_line = raw + "\t" + cleaned
                    out_file.write(error_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_directory", type=str)
    parser.add_argument("clean_directory", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()
    main(args)
