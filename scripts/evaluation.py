import sys

import project
from src.benchmark.benchmark import Benchmark, BenchmarkFiles, SUBSETS
from src.evaluation.evaluator import Evaluator
from src.evaluation.tolerant import tolerant_preprocess_sequences
from src.evaluation.print_methods import print_evaluator


def print_help():
    print("Usage:")
    print("    python3 demos/evaluation.py <benchmark> <subset> <file> [<original file>]")
    print()
    print("Arguments:")
    print("    <benchmark>: Choose a benchmark located at /external/benchmarks/.")
    print("    <subset>:    Specify 'development' or 'test'.")
    print("    <file>:      Name of the file with predicted sequences. "
          "This file must be located at /external/results/<benchmark>/<subset>/.")
    print()
    print("Example:")
    print("    python3 demos/evaluation.py Wiki test google.txt")


def get_arguments():
    n_args = len(sys.argv) - 1
    if n_args != 3:
        print("ERROR: please specify 3. Found %i." % n_args)
        exit(1)
    benchmark = sys.argv[1]
    subset = sys.argv[2]
    if subset not in ("development", "test"):
        print("ERROR: please specify subset from {development, test}.")
        exit(1)
    subset = SUBSETS[subset]
    file_name = sys.argv[3]
    return benchmark, subset, file_name


if __name__ == "__main__":
    if len(sys.argv) == 1 or "-h" in sys.argv or "-help" in sys.argv or "help" in sys.argv:
        print_help()
        exit(0)

    benchmark, subset, file_name = get_arguments()

    benchmark = Benchmark(benchmark, subset)
    correct_sequences = benchmark.get_sequences(BenchmarkFiles.CORRECT)
    corrupt_sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
    if file_name == "corrupt.txt":
        predicted_sequences = corrupt_sequences
    else:
        predicted_sequences = benchmark.get_predicted_sequences(file_name)
    original_sequences = correct_sequences

    evaluator = Evaluator()
    for seq_id, (original, correct, corrupt, predicted) in \
            enumerate(zip(original_sequences, correct_sequences, corrupt_sequences, predicted_sequences)):

        if benchmark.name == "acl" and original.startswith("#"):
            print(original)
            continue

        correct_processed, corrupt_processed, predicted_processed = \
            tolerant_preprocess_sequences(original, correct, corrupt, predicted)

        evaluator.evaluate(None,
                           None,
                           original_sequence=correct_processed,
                           corrupt_sequence=corrupt_processed,
                           predicted_sequence=predicted_processed,
                           evaluate_ed=False)
        print(original)
        print(corrupt)
        evaluator.print_sequence()
        print()

    print_evaluator(evaluator)
    evaluator.save_json(benchmark.name, subset.name.lower(), file_name)
