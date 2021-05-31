import json
import sys
import multiprocessing as mp

import project
from src.helper.files import read_lines, write_file
from src.spelling.evaluation import TokenErrorType, longest_common_subsequence
from spelling_evaluation_space_preference import get_token_edit_labels


def get_token_errors(sequence_id, ground_truth, corrupt_sequence):
    print(sequence_id)
    print(corrupt_sequence)
    return get_token_edit_labels(ground_truth, corrupt_sequence)


if __name__ == "__main__":
    benchmark = sys.argv[1]  # "arXiv.OCR.no_spaces"  #"ACL"
    set = "test" if "-test" in sys.argv else "development"
    n_sequences = 1000

    benchmark_dir = "/home/hertel/tokenization-repair-dumps/data/spelling/" + benchmark + "/" + set + "/"
    out_file = "/home/hertel/tr-adgit/spelling-evaluation-webapp/results/" + benchmark + "." + set + ".json"

    corrupt_paragraphs = read_lines(benchmark_dir + "corrupt.txt")[:n_sequences]
    spelling_paragraphs = read_lines(benchmark_dir + "spelling.txt")[:n_sequences]
    n_sequences = len(corrupt_paragraphs)

    if benchmark in ("ACL", "arXiv.OCR", "Wiki.typos.spaces", "Wiki.typos.no_spaces"):
        approaches = [
            # TODO "nastase",
            "google",
            #"ours+google",
            "ours+post+google",
            #"ours.new+post+google",
            "oracle+post+google"
        ]
    elif benchmark == "arXiv.OCR.punctuation":
        approaches = []
    elif benchmark == "test":
        approaches = [
            "test"
        ]
    elif benchmark == "arXiv.OCR.punctuation":
        approaches = []
    elif benchmark in ("Wiki.typos-split.spaces", "Wiki.typos-split.no_spaces"):
        approaches = [
            "google",
            "ours.new+post+google",
            "oracle+post+google",
        ]
    else:
        raise Exception("unknown benchmark '%s'" % benchmark)
    if benchmark == "ACL":
        approaches.append("nastase")
    #if benchmark == "arXiv.OCR":
    #    approaches.append("BID-the-one-from-paper")


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
