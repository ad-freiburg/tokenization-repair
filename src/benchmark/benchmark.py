from typing import Optional, List

from enum import Enum
import numpy as np

from src.settings import paths
from src.helper.files import read_lines, make_directory_recursive
from src.benchmark.subset import Subset


NOISE_LEVELS = [0, 0.1]
ERROR_PROBABILITIES = [0.1, 1, np.inf]

SUBSETS = {
    "tuning": Subset.TUNING,
    "development": Subset.DEVELOPMENT,
    "test": Subset.TEST
}


def get_subset(subset_name: str) -> Subset:
    return SUBSETS[subset_name]


class BenchmarkFiles(Enum):
    CORRECT = "correct.txt"
    CORRUPT = "corrupt.txt"
    INSERTIONS = "insertions.txt"
    DELETIONS = "deletions.txt"
    ORIGINAL = "original.txt"


class Benchmark:
    def __init__(self,
                 name: str,
                 subset: Subset,
                 subfolder: Optional[str] = None):
        self.name = name
        self.subset = subset
        self.subfolder = subfolder

    def _sub_directory(self):
        return paths.benchmark_sub_directory(self.name, self.subset, self.subfolder)

    def _benchmark_directory(self):
        directory = paths.BENCHMARKS_DIR + self._sub_directory()
        make_directory_recursive(directory)
        return directory

    def _results_directory(self):
        directory = paths.RESULTS_DIR + self._sub_directory()
        make_directory_recursive(directory)
        return directory

    def make_directories(self):
        self._benchmark_directory()
        self._results_directory()

    def get_file(self, file: BenchmarkFiles) -> str:
        return self._benchmark_directory() + file.value

    def get_sequences(self, file: BenchmarkFiles):
        path = self.get_file(file)
        sequences = read_lines(path)
        return sequences

    def get_sequence_pairs(self,
                           corrupt_file: BenchmarkFiles):
        correct_sequences = self.get_sequences(BenchmarkFiles.CORRECT)
        corrupt_sequences = self.get_sequences(corrupt_file)
        sequence_pairs = list(zip(correct_sequences, corrupt_sequences))
        return sequence_pairs

    def get_results_directory(self):
        return self._results_directory()

    def get_predicted_sequences(self, predicted_file: str):
        directory = self._results_directory()
        lines = read_lines(directory + predicted_file)
        try:
            float(lines[-1])
            sequences = lines[:-1]
        except:
            sequences = lines
        return sequences

    def get_mean_runtime(self, predicted_file: str) -> float:
        lines = self.get_predicted_sequences(predicted_file)
        try:
            runtime = float(lines[-1])
            mean_runtime = runtime / (len(lines) - 1)
        except ValueError:
            mean_runtime = 0
        return mean_runtime


def get_benchmark_name(noise_level: float,
                       p: float) -> str:
    noise_str = ("%.1f" % noise_level) if noise_level > 0 else "0"
    if p == np.inf:
        p_str = "inf"
    elif p == 1:
        p_str = "1"
    else:
        p_str = "%.1f" % p
    name = "%s_%s" % (noise_str, p_str)
    return name


def get_benchmark(noise_level: float,
                  p: float,
                  subset: Subset) -> Benchmark:
    name = get_benchmark_name(noise_level, p)
    return Benchmark(name, subset)


def get_error_probabilities():
    return ERROR_PROBABILITIES


def all_benchmarks(subset: Subset) -> List[Benchmark]:
    benchmarks = []
    for n in NOISE_LEVELS:
        for p in ERROR_PROBABILITIES:
            benchmarks.append(get_benchmark(n, p, subset))
    return benchmarks
