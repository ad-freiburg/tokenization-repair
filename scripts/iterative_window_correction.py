import project
from src.interactive.parameters import ParameterGetter, Parameter

params = [
    Parameter("model", "-m", "str",
              help_message="Choose a model out of {combined, combined_robust, softmax, softmax_robust, sigmoid, "
                           "sigmoid_robust}."),
    Parameter("benchmarks", "-b", "str",
              dependencies=[
                  ("0", [Parameter("noise_type", "-n", "str",
                                   help_message="Choose the insertion and deletion thresholds tuned on a benchmark.\n"
                                                "Specify the benchmark as {TYPOS}_{TOKEN-ERRORS}, where TYPOS is one "
                                                "out of {0, 0.1, 0.2} and TOKEN_ERRORS is one out of {0.1, 0.2, 0.3, "
                                                "0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, inf} (inf stands for the benchmark "
                                                "with all spaces removed).\n"
                                                "For example 0_0.5 for no typos and 50% "
                                                "tokenization error rate, or 0.2_inf for 20% typo rate and no spaces.")
                         ]
                   )
              ]
              ),
    Parameter("test", "-t", "boolean"),
    Parameter("continue", "-c", "boolean")
]
getter = ParameterGetter(params)
parameters = getter.get()


from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.corrector.iterative_window_corrector import IterativeWindowCorrector
from src.corrector.threshold_holder import ThresholdHolder, FittingMethod
from src.load.load_char_lm import load_default_char_lm
from src.interactive.sequence_generator import interactive_sequence_generator
from src.helper.time import time_diff, timestamp
from src.evaluation.predictions_file_writer import PredictionsFileWriter


if __name__ == "__main__":
    model = load_default_char_lm(parameters["model"])
    space_index = model.get_encoder().encode_char(' ')
    threshold_holder = ThresholdHolder(fitting_method=FittingMethod.SINGLE_RUN)
    window_size = 10

    benchmark_names = parameters["benchmarks"]
    if not isinstance(benchmark_names, list):
        benchmark_names = [benchmark_names]

    for b_i, benchmark_name in enumerate(benchmark_names):
        if benchmark_name == "0":
            sequences = interactive_sequence_generator()
            file_writer = None
        else:
            subset = Subset.TEST if parameters["test"] else Subset.DEVELOPMENT
            benchmark = Benchmark(benchmark_name,
                                  subset=subset)
            sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
            file_writer = PredictionsFileWriter(benchmark.get_results_directory() + parameters["model"] + ".txt")
            if b_i == 0 and parameters["continue"]:
                file_writer.load()

        if benchmark_name == "0":
            noise_type = parameters["noise_type"]
        else:
            noise_type = benchmark_name
        if noise_type.endswith("_inf"):
            insertion_benchmark_name = noise_type[:-3] + "0.1"
            deletion_benchmark_name = noise_type[:-3] + "1"
            insertion_threshold, _ = threshold_holder.get_thresholds(model_name=parameters["model"],
                                                                     noise_type=insertion_benchmark_name)
            _, deletion_threshold = threshold_holder.get_thresholds(model_name=parameters["model"],
                                                                    noise_type=deletion_benchmark_name)
        else:
            insertion_threshold, deletion_threshold = threshold_holder.get_thresholds(model_name=parameters["model"],
                                                                                      noise_type=noise_type)

        corrector = IterativeWindowCorrector(model,
                                             insertion_threshold=insertion_threshold,
                                             deletion_threshold=deletion_threshold,
                                             window_size=window_size,
                                             verbose=benchmark_name == "0")

        for s_i, sequence in enumerate(sequences):
            if file_writer is not None and s_i < file_writer.n_sequences():
                continue
            if benchmark_name.endswith("_inf"):
                sequence = ' '.join(sequence)
            print(sequence)
            start_time = timestamp()
            predicted = corrector.correct(sequence)
            runtime = time_diff(start_time)
            print(predicted)
            if file_writer is not None:
                file_writer.add(predicted, runtime)
                file_writer.save()
