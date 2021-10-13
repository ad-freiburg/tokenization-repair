import project
from src.interactive.parameters import ParameterGetter, Parameter


params = [Parameter("n_tokens", "-n", "int"),
          Parameter("benchmark", "-b", "str"),
          Parameter("test", "-t", "boolean")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.corrector.baselines.maximum_matching_corrector import MaximumMatchingCorrector
from src.interactive.sequence_generator import interactive_sequence_generator
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp

if __name__ == "__main__":
    n_tokens = parameters["n_tokens"]
    corrector = MaximumMatchingCorrector(n=n_tokens)

    benchmark_name = parameters["benchmark"]
    if benchmark_name == "0":
        sequences = interactive_sequence_generator()
        file_writer = None
    else:
        subset = Subset.TEST if parameters["test"] else Subset.DEVELOPMENT
        benchmark = Benchmark(benchmark_name, subset)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        file = benchmark.get_results_directory() + "maximum_matching_%i.txt" % n_tokens
        file_writer = PredictionsFileWriter(file)

    for sequence in sequences:
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        print(runtime)
        if file_writer is not None:
            file_writer.add(predicted, runtime)

    if file_writer is not None:
        file_writer.save()
