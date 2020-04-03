import sys

import project
from src.evaluation.evaluator import Evaluator
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles, get_benchmark_name, get_error_probabilities
from src.evaluation.results_holder import ResultsHolder, Metric


NOISE_LEVELS = [0, 0.1, 0.2]
ERROR_PROBABILITIES = get_error_probabilities()


if __name__ == "__main__":
    file_name = sys.argv[1]
    approach_name = sys.argv[2]

    results_holder = ResultsHolder()

    for noise_level in NOISE_LEVELS:
        for p in ERROR_PROBABILITIES:
            benchmark_name = get_benchmark_name(noise_level, p)
            benchmark_subset = Subset.TEST
            print(benchmark_name)
            benchmark = Benchmark(benchmark_name, benchmark_subset)
            sequence_pairs = benchmark.get_sequence_pairs(BenchmarkFiles.CORRUPT)

            if file_name == "corrupt.txt":
                predicted_sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
                mean_runtime = 0
            else:
                try:
                    predicted_sequences = benchmark.get_predicted_sequences(file_name)[:len(sequence_pairs)]
                    mean_runtime = benchmark.get_mean_runtime(file_name)
                except FileNotFoundError:
                    predicted_sequences = []
                    mean_runtime = 0

            if len(predicted_sequences) == len(sequence_pairs):
                evaluator = Evaluator()

                for i, (correct, corrupt) in enumerate(sequence_pairs):
                    predicted = predicted_sequences[i]
                    evaluator.evaluate(file_name=None,
                                       line=None,
                                       original_sequence=correct,
                                       corrupt_sequence=corrupt,
                                       predicted_sequence=predicted,
                                       evaluate_ed=False)
                f1 = evaluator.f1()
                acc = evaluator.sequence_accuracy()
                print("f1  = %2.2f" % (f1 * 100))
                print("acc = %2.2f" % (acc * 100))
                print("t   = %.2f" % mean_runtime)
            else:
                f1 = acc = 0
            metric_value_pairs = [(Metric.F1, f1),
                                  (Metric.SEQUENCE_ACCURACY, acc),
                                  (Metric.MEAN_RUNTIME, mean_runtime)]
            results_holder.set(benchmark_name, benchmark_subset, approach_name, metric_value_pairs)

    results_holder.save()
