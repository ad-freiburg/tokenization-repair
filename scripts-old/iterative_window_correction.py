import project
from src.interactive.parameters import ParameterGetter, Parameter

params = [
    Parameter("model", "-m", "str",
              help_message="Choose a model out of {combined, combined_robust, softmax, softmax_robust, sigmoid, "
                           "sigmoid_robust}."),
    Parameter("benchmark", "-b", "str",
              dependencies=[("0", [Parameter("noise_type", "-n", "str")])]),
    Parameter("test", "-t", "boolean"),
    Parameter("two_pass", "-tp", "str"),
    Parameter("continue", "-c", "boolean")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.benchmark.two_pass_benchmark import TwoPassBenchmark
from src.corrector.iterative_window_corrector import IterativeWindowCorrector
from src.corrector.threshold_holder import ThresholdHolder, FittingMethod
from src.load.load_char_lm import load_default_char_lm
from src.interactive.sequence_generator import interactive_sequence_generator
from src.helper.time import time_diff, timestamp
from src.evaluation.predictions_file_writer import PredictionsFileWriter


if __name__ == "__main__":
    model = load_default_char_lm(parameters["model"])
    space_index = model.get_encoder().encode_char(' ')
    window_size = 10

    benchmark_name = parameters["benchmark"]

    if benchmark_name == "0":
        sequences = interactive_sequence_generator()
        file_writer = None
        threshold_holder = ThresholdHolder(fitting_method=FittingMethod.SINGLE_RUN)
    else:
        subset = Subset.TEST if parameters["test"] else Subset.DEVELOPMENT
        if parameters["two_pass"] == "0":
            benchmark = Benchmark(benchmark_name,
                                  subset=subset)
            threshold_holder = ThresholdHolder(fitting_method=FittingMethod.SINGLE_RUN)
        else:
            benchmark = TwoPassBenchmark(benchmark_name, parameters["two_pass"], subset)
            threshold_holder = ThresholdHolder(fitting_method=FittingMethod.TWO_PASS)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        file_writer = PredictionsFileWriter(benchmark.get_results_directory()
                                            + ("two_pass_" if parameters["two_pass"] != "0" else "")
                                            + parameters["model"] + ".txt")
        if parameters["continue"]:
            file_writer.load()

    if benchmark_name == "0":
        noise_type = parameters["noise_type"]
    else:
        noise_type = benchmark_name
    if parameters["two_pass"] == "0" and noise_type.endswith("_inf"):
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
        if benchmark_name.endswith("_inf") and parameters["two_pass"] == "0":
            sequence = ' '.join(sequence)
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        if parameters["two_pass"] == "0" or predicted != sequence:
            print(sequence)
            print(predicted)
        if file_writer is not None:
            file_writer.add(predicted, runtime)
            if parameters["two_pass"] == "0":
                file_writer.save()

    if file_writer is not None:
        file_writer.save()
