import json

import project
from src.helper.files import read_lines, write_file
from src.spelling.evaluation import TokenErrorType, get_ground_truth_labels, longest_common_subsequence


if __name__ == "__main__":
    set = "development"
    start = 0
    end = 319

    benchmark_dir = "/home/hertel/tokenization-repair-dumps/data/spelling/ACL/" + set + "/"

    corrupt_paragraphs = read_lines(benchmark_dir + "corrupt.txt")
    spelling_paragraphs = read_lines(benchmark_dir + "spelling.txt")

    approaches = [
        "google",
        "TextRazor",
        "the_one",
        "the_one+google",
        "the_one+postprocessing",
        "the_one+postprocessing+google",
        "the_one+postprocessing+TextRazor",
        "gold+google",
        "hunspell",
        "the_one+hunspell",
        "gold+hunspell"
    ]

    predicted_sequences = {
        approach: read_lines(benchmark_dir + approach + ".txt") for approach in approaches
    }

    error_counts = {error: 0 for error in TokenErrorType}
    correct_prediction_counts = {approach: {error: 0 for error in TokenErrorType} for approach in approaches}
    sequences_data = []

    for i, (corrupt, spelling) in \
            enumerate(zip(corrupt_paragraphs, spelling_paragraphs)):
        if i < start or i > end:
            continue
        errors = get_ground_truth_labels(spelling, corrupt)
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
        print(correct_prediction_counts[approach])

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
    write_file(benchmark_dir + "results.json", json.dumps(data))
