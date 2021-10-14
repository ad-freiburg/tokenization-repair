# -*- coding: future_fstrings -*-

import json
import multiprocessing as mp
import argparse

import project
from src.helper.files import read_lines, write_file
from src.spelling.evaluation import TokenErrorType, longest_common_subsequence
from src.spelling.space_preference import get_token_edit_labels
from src.settings import paths


def get_token_errors(sequence_id, ground_truth, corrupt_sequence):
    print(sequence_id)
    print(corrupt_sequence)
    return get_token_edit_labels(ground_truth, corrupt_sequence)


def main(args):
    benchmark = args.benchmark
    set = "test" if args.test else "development"
    n_sequences = args.n_sequences
    approaches = args.approaches

    benchmark_dir = paths.DUMP_DIR + "spelling/" + benchmark + "/" + set + "/"
    out_file = paths.DUMP_DIR + "spelling/" + benchmark + "." + set + ".json"

    corrupt_paragraphs = read_lines(benchmark_dir + "corrupt.txt")[:n_sequences]
    spelling_paragraphs = read_lines(benchmark_dir + "spelling.txt")[:n_sequences]

    predicted_sequences = {
        approach: read_lines(benchmark_dir + approach + ".txt") for approach in approaches
    }

    error_counts = {error: 0 for error in TokenErrorType}
    correct_prediction_counts = {approach: {error: 0 for error in TokenErrorType} for approach in approaches}
    sequences_data = []

    n_processes = max(mp.cpu_count() - 1, 1)
    with mp.Pool(n_processes) as pool:
        sequence_errors = pool.starmap(get_token_errors,
                                       list(zip(range(len(spelling_paragraphs)),
                                                spelling_paragraphs,
                                                corrupt_paragraphs)))

    for i, (corrupt, spelling) in \
            enumerate(zip(corrupt_paragraphs, spelling_paragraphs)):
        print("sequence", i)
        print(spelling)
        print(corrupt)
        #errors = get_token_edit_labels(spelling, corrupt)
        errors = sequence_errors[i]
        for error in errors:
            error_counts[error] += 1
        correct_tokens = spelling.split()
        corrupt_tokens = corrupt.split()
        matched = longest_common_subsequence(correct_tokens, corrupt_tokens)
        matched_corrupt_tokens = {n for m, n in matched}
        corrupt_labels = ["NONE" if j in matched_corrupt_tokens else "WRONG" for j in range(len(corrupt_tokens))]
        sequence_data = {
            "corrupt": {
                "tokens": corrupt_tokens,
                "labels": corrupt_labels,
            },
            "correct": {
                "tokens": correct_tokens,
                "labels": [error.name for error in errors]
            },
            "predicted": {}
        }
        for approach in approaches:
            predicted = predicted_sequences[approach][i]
            predicted_tokens = predicted.split()
            matched = longest_common_subsequence(correct_tokens, predicted_tokens)
            labels = ["WRONG" for _ in predicted_tokens]
            for m, n in matched:
                correct_prediction_counts[approach][errors[m]] += 1
                labels[n] = errors[m]
            sequence_data["predicted"][approach] = {
                "tokens": predicted_tokens,
                "labels": [label.name if isinstance(label, TokenErrorType) else label for label in labels]
            }
        sequences_data.append(sequence_data)

    total_errors = sum(error_counts[error] for error in TokenErrorType if error != TokenErrorType.NONE)
    print(error_counts)
    for approach in approaches:
        print(approach, correct_prediction_counts[approach])

    for approach in approaches:
        print("\n" + approach)
        counts = correct_prediction_counts[approach]
        for error_type in TokenErrorType:
            print(f"{error_type.name}: {counts[error_type]}/{error_counts[error_type]}")
        total_correct = sum(counts[error] for error in TokenErrorType if error != TokenErrorType.NONE)
        print(f"all: {total_correct}/{total_errors} ({total_correct / total_errors * 100:.2f}%)")

    data = {
        "total": {error.name: error_counts[error] for error in TokenErrorType},
        "correct": {approach: {error.name: correct_prediction_counts[approach][error] for error in TokenErrorType}
                    for approach in approaches},
        "sequences": sequences_data
    }
    write_file(out_file, json.dumps(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the spelling evaluation."
                                                 "A JSON with the results will be saved at "
                                                 "/external/spelling/.")
    parser.add_argument("benchmark", type=str,
                        help="Name of the benchmark. The benchmark must be located at /external/spelling.")
    parser.add_argument("--test", action="store_true", required=False,
                        help="Set this flag to use the test set of the benchmark (default: development set).")
    parser.add_argument("-n", dest="n_sequences", type=int, default=1000,
                        help="(Maximum) number of sequences used for the evaluation (default: 1000).")
    parser.add_argument("-a", dest="approaches", type=str, nargs="+",
                        default=["google", "ours+google", "oracle+google"],
                        help="Specify a list of approaches to be evaluated. The approach name must be equal to a file "
                             "name in /external/spelling/<benchmark>/<set>/, but without the '.txt' ending. "
                             "Default: google ours+google oracle+google")
    args = parser.parse_args()
    main(args)
