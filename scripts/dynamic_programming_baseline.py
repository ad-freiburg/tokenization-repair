import project
from src.interactive.parameters import ParameterGetter, Parameter


params = [Parameter("bigram_postprocessing", "-bi", "boolean"),
          Parameter("interactive", "-i", "boolean"),
          Parameter("test", "-t", "boolean"),
          Parameter("file_name", "-f", "str"),
          Parameter("verbose", "-v", "boolean")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import numpy as np

from src.interactive.sequence_generator import interactive_sequence_generator
from src.baselines.dynamic_programming import DynamicProgrammingCorrector
from src.benchmark.benchmark import Subset, BenchmarkFiles, NOISE_LEVELS, get_benchmark
from src.helper.time import time_diff, timestamp
from src.evaluation.predictions_file_writer import PredictionsFileWriter


if __name__ == "__main__":
    corrector = DynamicProgrammingCorrector(bigram_postprocessing=parameters["bigram_postprocessing"])

    if parameters["interactive"]:
        benchmarks = [None]
    else:
        subset = Subset.TEST if parameters["test"] else Subset.DEVELOPMENT
        benchmarks = [get_benchmark(noise_level, np.inf, subset) for noise_level in NOISE_LEVELS]

    for benchmark in benchmarks:
        if benchmark is None:
            sequences = interactive_sequence_generator()
            file_writer = None
        else:
            print(benchmark.name)
            sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
            file_writer = PredictionsFileWriter(benchmark.get_results_directory() + parameters["file_name"])

        for sequence in sequences:
            if parameters["verbose"]:
                print(sequence)

            start = timestamp()
            predicted = corrector.correct(sequence)
            runtime = time_diff(start)

            if benchmark is None or parameters["verbose"]:
                print(predicted)
            else:
                file_writer.add(predicted, runtime)

        if file_writer is not None:
            file_writer.save()
