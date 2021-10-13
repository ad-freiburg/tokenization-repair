import sys

import project
from src.benchmark.benchmark import Benchmark, SUBSETS, BenchmarkFiles, Subset, all_benchmarks
from src.evaluation.evaluator import Evaluator
from src.datasets.wikipedia import Wikipedia
from src.helper.data_structures import izip
from src.evaluation.print_methods import print_evaluator
from src.helper.files import read_sequences
from src.settings import paths
from src.evaluation.tolerant import tolerant_preprocess_sequences


def table_row(f1_list, acc_list):
    row = ""
    for value in f1_list + acc_list:
        row += "&   %.2f " % (value * 100)
    return row


if __name__ == "__main__":
    benchmark_name = sys.argv[1]
    subset = SUBSETS[sys.argv[2]]
    predictions_file_name = sys.argv[3]
    n_sequences = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    eval_all = benchmark_name == "all"
    if eval_all:
        benchmarks = all_benchmarks(subset)
    else:
        benchmarks = [Benchmark(benchmark_name, subset)]

    f1_list = []
    acc_list = []

    for benchmark in benchmarks:
        original_sequences = {Subset.TUNING: read_sequences(paths.WIKI_TUNING_SENTENCES),
                              Subset.DEVELOPMENT: Wikipedia.development_sequences(),
                              Subset.TEST: Wikipedia.test_sequences()}[subset]

        sequence_pairs = benchmark.get_sequence_pairs(BenchmarkFiles.CORRUPT)
        if predictions_file_name == "corrupt.txt":
            predicted_sequences = [corrupt for _, corrupt in sequence_pairs]
        else:
            predicted_sequences = benchmark.get_predicted_sequences(predictions_file_name)

        evaluator = Evaluator()

        for s_i, original, (correct, corrupt), predicted in izip(original_sequences, sequence_pairs, predicted_sequences):
            if s_i == n_sequences:
                break

            correct_processed, corrupt_processed, predicted_processed = \
                tolerant_preprocess_sequences(original, correct, corrupt, predicted)

            evaluator.evaluate(predictions_file_name,
                               s_i,
                               original_sequence=correct_processed,
                               corrupt_sequence=corrupt_processed,
                               predicted_sequence=predicted_processed,
                               evaluate_ed=False)

            if not eval_all:
                print(original)
                print(correct)
                print(corrupt)
                print(predicted)
                print(correct_processed)
                print(corrupt_processed)
                print(predicted_processed)
                evaluator.print_sequence()
                print()

        f1, acc = print_evaluator(evaluator)

        f1_list.append(f1)
        acc_list.append(acc)

    print(f1_list)
    print(acc_list)
    print(table_row(f1_list, acc_list))
