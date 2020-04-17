from enchant.checker import SpellChecker

import project
from src.interactive.sequence_generator import interactive_sequence_generator
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp


def is_split(original: str, split: str):
    return split.replace(' ', '') == original


class PyEnchantTokenizationCorrector:
    def __init__(self):
        self.checker = SpellChecker("en")

    def correct(self, sequence: str) -> str:
        self.checker.set_text(sequence)
        for error in self.checker:
            for suggestion in self.checker.suggest():
                if is_split(error.word, suggestion):
                    self.checker.replace(suggestion)
                    break
        return self.checker.get_text()


if __name__ == "__main__":
    import sys

    corrector = PyEnchantTokenizationCorrector()

    for benchmark_name in sys.argv[1:]:
        benchmark = Benchmark(benchmark_name, Subset.DEVELOPMENT)
        print(benchmark.name)
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + "enchant.txt")
        for s_i, sequence in enumerate(benchmark.get_sequences(BenchmarkFiles.CORRUPT)):
            print(s_i)
            start_time = timestamp()
            predicted = corrector.correct(sequence)
            runtime = time_diff(start_time)
            file_writer.add(predicted, runtime)
        file_writer.save()
