import project
from src.baselines.left_to_right_baseline import LeftToRightCorrector
from src.benchmark.benchmark import all_benchmarks, Subset, BenchmarkFiles
from src.helper.time import time_diff, timestamp
from src.evaluation.predictions_file_writer import PredictionsFileWriter


if __name__ == "__main__":
    corrector = LeftToRightCorrector()

    for benchmark in all_benchmarks(Subset.TEST):
        print(benchmark.name)
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + "greedy.txt")
        for sequence in benchmark.get_sequences(BenchmarkFiles.CORRUPT):
            start_time = timestamp()
            predicted = corrector.correct(sequence)
            runtime = time_diff(start_time)
            file_writer.add(predicted, runtime)
        file_writer.save()
