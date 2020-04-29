import sys

import project
from src.settings import paths
from src.helper.files import read_lines
from src.benchmark.benchmark import get_benchmark, ERROR_PROBABILITIES, Subset, BenchmarkFiles

if __name__ == "__main__":
    file_name = sys.argv[1]

    for p in ERROR_PROBABILITIES:
        benchmark = get_benchmark(0, p, Subset.DEVELOPMENT)
        print("== %s ==" % benchmark.name)
        path = benchmark.get_results_directory() + file_name
        total_runtime = float(read_lines(path)[-1])
        mean_runtime = total_runtime / 10000
        print("mean = %.2f" % mean_runtime)
        n_chars = sum(len(sequence) for sequence in benchmark.get_sequences(BenchmarkFiles.CORRUPT))
        per_chars = 1000
        normalized_runtime = total_runtime / n_chars * per_chars
        print("normalized(%i chars) = %.2f" % (per_chars, normalized_runtime))