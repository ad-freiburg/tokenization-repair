import sys

import project
from src.spelling.evaluation import TokenErrorType, longest_common_subsequence
from src.helper.files import read_lines
from src.spelling.space_preference import edit_distance, get_labels


if __name__ == "__main__":
    if len(sys.argv) == 1:
        corrupt_sequences = ["w h i c hi m p l i e st h a t", "hello*", "this", "this\is", "this i f a test",
                             "Letghe"]
        correct_sequences = ["which implies that", "hello", "thus", "this is", "this is a test", "Let the"]
    else:
        n = int(sys.argv[3])
        corrupt_sequences = read_lines(sys.argv[1])[:n]
        correct_sequences = read_lines(sys.argv[2])[:n]

    label_counts = {label: 0 for label in TokenErrorType}

    for correct, corrupt in zip(correct_sequences, corrupt_sequences):
        #d, matrix = levenshtein(corrupt, correct, return_matrix=True, substitutions=False)
        #print(d)
        #operations = get_operations(corrupt, correct, matrix, substitutions=False)
        #for op in operations:
        #    print(op)
        #labels = get_ground_truth_labels(correct, corrupt)
        #for token, label in zip(correct.split(), labels):
        #    print(token, label)
        d, operations = edit_distance(correct, corrupt)
        print(d, operations)
        labels = get_labels(correct, operations)
        print(corrupt)
        for token, label in zip(correct.split(), labels):
            print(token, label)
            label_counts[label] += 1

    print()
    for label in TokenErrorType:
        print(label, label_counts[label])