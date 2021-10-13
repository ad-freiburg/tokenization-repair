import sys

from project import src
from src.datasets.wikipedia import Wikipedia
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.sequence.pdf2text_token_corruptor import Pdf2textTokenCorruptor
from src.helper.files import open_file


if __name__ == "__main__":
    BENCHMARK_NAME = "pdf2text"

    development_sequences = Wikipedia.development_sequences()
    test_sequences = Wikipedia.test_sequences()

    development_benchmark = Benchmark(BENCHMARK_NAME, Subset.DEVELOPMENT)
    test_benchmark = Benchmark(BENCHMARK_NAME, Subset.TEST)

    corruptor = Pdf2textTokenCorruptor()

    for benchmark, true_sequences in ((development_benchmark, development_sequences),
                                      (test_benchmark, test_sequences)):
        correct_file = open_file(benchmark.get_file(BenchmarkFiles.CORRECT))
        corrupt_file = open_file(benchmark.get_file(BenchmarkFiles.CORRUPT))
        for true_sequence in true_sequences:
            corrupt_sequence = corruptor.corrupt(true_sequence)
            correct_file.write(true_sequence + '\n')
            corrupt_file.write(corrupt_sequence + '\n')
        correct_file.close()
        corrupt_file.close()
