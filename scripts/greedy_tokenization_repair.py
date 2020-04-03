import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("model_type", "-type", "str",
              help_message="Choose from {fwd, bwd, bidir, combined, softmax_old, sigmoid_old, combined_old}.",
              dependencies=[
                  ("combined", [Parameter("bwd_model_name", "-bwd", "str")]),
                  ("combined_old", [Parameter("bwd_model_name", "-bwd", "str")])
              ]),
    Parameter("model_name", "-name", "str"),
    Parameter("noise_type", "-noise", "str"),
    Parameter("benchmark", "-benchmark", "str"),
    Parameter("out_file", "-f", "str"),
    Parameter("initialize", "-init", "boolean")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.load.load_char_lm import load_char_lm
from src.corrector.threshold_holder import ThresholdHolder
from src.corrector.greedy_corrector import GreedyCorrector
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.interactive.sequence_generator import interactive_sequence_generator
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import timestamp, time_diff


if __name__ == "__main__":
    # load model:
    model_type = parameters["model_type"]
    model_name = parameters["model_name"]
    bwd_model_name = parameters["bwd_model_name"] if "bwd_model_name" in parameters else None
    model = load_char_lm(model_type, model_name, bwd_model_name)

    # load thresholds:
    threshold_holder = ThresholdHolder()
    if parameters["model_type"] in ("combined", "combined_old"):
        insertion_threshold, deletion_threshold = threshold_holder.get_thresholds(
            fwd_model_name=model_name,
            bwd_model_name=bwd_model_name,
            noise_type=parameters["noise_type"]
        )
    else:
        insertion_threshold, deletion_threshold = threshold_holder.get_thresholds(
            model_name=model_name,
            noise_type=parameters["noise_type"]
        )

    # make greedy corrector:
    corrector = GreedyCorrector(model,
                                insertion_threshold=insertion_threshold,
                                deletion_threshold=deletion_threshold)

    # load benchmark sequences or interactive:
    benchmark_name = parameters["benchmark"]
    if benchmark_name == "0":
        benchmark = None
        sequences = interactive_sequence_generator()
    else:
        benchmark = Benchmark(benchmark_name,
                              subset=Subset.TEST if benchmark_name == "project" else Subset.DEVELOPMENT)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)

    # output file:
    if benchmark is None:
        file_writer = None
    else:
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + parameters["out_file"])
        if not parameters["initialize"]:
            file_writer.load()

    # iterate over sequences:
    total_runtime = 0
    n_sequences = 0
    for s_i, sequence in enumerate(sequences):
        if file_writer is not None and s_i < file_writer.n_sequences():
            continue
        print("sequence %i" % n_sequences)
        n_sequences += 1
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        total_runtime += runtime
        print(predicted)

        if file_writer is not None:
            file_writer.add(predicted, runtime)
            file_writer.save()

        print("total runtime = %.4f sec" % total_runtime)
        print("average runtime = %.4f sec" % (total_runtime / n_sequences))
