import sys

import project
from src.benchmark.benchmark import Subset, get_error_probabilities, get_benchmark


NOISE_LEVELS = [0, 0.1, 0.2]


if __name__ == "__main__":
    file = sys.argv[1]

    for noise_level in NOISE_LEVELS:
        for p in get_error_probabilities():
            benchmark = get_benchmark(noise_level, p, Subset.TEST)
            try:
                predicted = benchmark.get_predicted_sequences(file)[:-1]
            except FileNotFoundError:
                predicted = []
            progress = len(predicted) // 200
            benchmark_name = benchmark.name[:7]
            benchmark_name += ' ' * (7 - len(benchmark_name))
            print(benchmark_name, "|" * progress)
