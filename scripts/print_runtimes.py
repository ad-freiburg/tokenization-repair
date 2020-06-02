import numpy as np

import sys

import project
from src.helper.files import read_lines, file_exists
from src.benchmark.benchmark import all_benchmarks, ERROR_PROBABILITIES, Subset, BenchmarkFiles

if __name__ == "__main__":
    file_name = sys.argv[1]
    per_chars = 1000

    t_mean = []
    t_normalized = []

    for benchmark in all_benchmarks(Subset.TEST):
        print("== %s ==" % benchmark.name)
        path = benchmark.get_results_directory() + file_name
        total_runtime = float(read_lines(path)[-1]) if file_exists(path) else 0
        mean_runtime = total_runtime / 10000
        t_mean.append(mean_runtime)
        print("mean = %.2f" % mean_runtime)
        n_chars = sum(len(sequence) for sequence in benchmark.get_sequences(BenchmarkFiles.CORRUPT))
        normalized_runtime = total_runtime / n_chars * per_chars
        t_normalized.append(normalized_runtime)
        print("normalized(%i chars) = %.2f" % (per_chars, normalized_runtime))

    print("== total ==")
    print("mean = %.2f" % np.mean(t_mean))
    print("normalized(%i chars) = %.2f" % (per_chars, np.mean(t_normalized)))
