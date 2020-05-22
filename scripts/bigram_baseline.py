import sys

import project
from src.baselines.bigram_dynamic_corrector import BigramDynamicCorrector
from src.interactive.sequence_generator import interactive_sequence_generator
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.helper.time import time_diff, timestamp
from src.evaluation.predictions_file_writer import PredictionsFileWriter


if __name__ == "__main__":
    if len(sys.argv) == 1:
        benchmark_names = [None]
    else:
        benchmark_names = sys.argv[1:]

    corrector = BigramDynamicCorrector()

    for benchmark_name in benchmark_names:
        if benchmark_name is None:
            sequences = interactive_sequence_generator()
            file_writer = None
        else:
            benchmark = Benchmark(benchmark_name, Subset.DEVELOPMENT)
            sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
            file_writer = PredictionsFileWriter(benchmark.get_results_directory() + "bigrams.txt")

        for sequence in sequences:
            start_time = timestamp()
            predicted = corrector.correct(sequence)
            runtime = time_diff(start_time)
            print(predicted)
            if file_writer is not None:
                file_writer.add(predicted, runtime)

        if file_writer is not None:
            file_writer.save()
