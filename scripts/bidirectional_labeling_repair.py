import sys

import project
from src.corrector.labeling.labeling_corrector import LabelingCorrector
from src.interactive.sequence_generator import interactive_sequence_generator
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.benchmark.benchmark import Benchmark, BenchmarkFiles, SUBSETS
from src.helper.time import time_diff, timestamp
from src.corrector.threshold_holder import FittingMethod, ThresholdHolder


if __name__ == "__main__":
    name = sys.argv[1]

    benchmark_name = sys.argv[2]
    subset = SUBSETS[sys.argv[3]]
    holder = ThresholdHolder(FittingMethod.LABELING)
    insertion_threshold, deletion_threshold = holder.get_thresholds(name, noise_type=benchmark_name)

    corrector = LabelingCorrector(name, insertion_threshold, deletion_threshold)

    if benchmark_name is None:
        sequences = interactive_sequence_generator()
        file_writer = None
    else:
        benchmark = Benchmark(benchmark_name, subset)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + name + ".txt")

    for sequence in sequences:
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        if file_writer is not None:
            file_writer.add(predicted, runtime)

    if file_writer is not None:
        file_writer.save()
