from typing import Optional

import sys

import project
from src.helper.files import get_files, read_file, write_lines
from src.benchmark.benchmark import ERROR_PROBABILITIES, get_benchmark, Subset


BASE_DIR = "/nfs/students/mostafa-mohamed/paper_v2/benchmark_dumps/"
NOISE_LEVELS = [0, 0.2]


def get_predicted_sequences_dir(approach_prefix: str, benchmark_name: str) -> Optional[str]:
    sub_dirs = get_files(BASE_DIR)
    approach_benchmark_prefix = approach_prefix + benchmark_name
    for subdir in sub_dirs:
        if subdir.startswith(approach_benchmark_prefix):
            return BASE_DIR + subdir + "/fixed/"
    return None


if __name__ == "__main__":
    N = 10000
    if sys.argv[1] == "dp":
        approach_prefix = "dp_fixer_wikipedia_"
        out_file = "dp_fixer.txt"
    else:
        approach_prefix = "bicontext_fixer_wikipedia-"
        out_file = "bicontext_fixer.txt"
    for noise_level in NOISE_LEVELS:
        for p in ERROR_PROBABILITIES:
            benchmark = get_benchmark(noise_level, p, Subset.TEST)
            print(benchmark.name)
            predictions_dir = get_predicted_sequences_dir(approach_prefix, benchmark.name)
            print(" results dir: %s" % str(predictions_dir))
            if predictions_dir is not None:
                predictions_files = get_files(predictions_dir)
                predictions_files = [file_name for file_name in predictions_files if file_name.endswith(".txt")]
                n_sequences = len(predictions_files)
                print(" %i sequence files" % n_sequences)
                if n_sequences == N:
                    predicted_sequences = []
                    for file_name in sorted(predictions_files):
                        sequence = read_file(predictions_dir + file_name)
                        predicted_sequences.append(sequence)
                    out_path = benchmark.get_results_directory() + out_file
                    write_lines(out_path, predicted_sequences)
                    print(" written to %s" % out_path)
