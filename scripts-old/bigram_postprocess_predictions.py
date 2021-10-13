import sys

import project
from src.postprocessing.bigram import BigramPostprocessor
from src.benchmark.benchmark import Benchmark, Subset
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp


if __name__ == "__main__":
    benchmark_name = sys.argv[1]
    in_file_name = sys.argv[2]
    out_file_name = sys.argv[3]

    corrector = BigramPostprocessor()
    benchmark = Benchmark(benchmark_name, Subset.DEVELOPMENT)
    file_writer = PredictionsFileWriter(benchmark.get_results_directory() + out_file_name)

    for sequence in benchmark.get_predicted_sequences(in_file_name):
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        if predicted != sequence:
            print(predicted)
        file_writer.add(predicted, runtime)

    file_writer.save()

