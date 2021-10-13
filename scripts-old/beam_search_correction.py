from project import src
from src.interactive.parameters import ParameterGetter, Parameter


params = [
    Parameter("benchmark", "-benchmark", "str",
              help_message="Name of the benchmark."),
    Parameter("n_sequences", "-n", "int"),
    Parameter("model_name", "-model", "str",
              help_message="Name of the unidirectional model to be used."),
    Parameter("backward", "-bwd", "boolean",
              help_message="Set 1 if the model is a backward model."),
    Parameter("n_beams", "-b", "int",
              help_message="Number of beams for beam search."),
    Parameter("average_log_likelihood", "-avg", "boolean",
              help_message="Set 1 to divide beam scores by sequence length."),
    Parameter("penalty", "-p", "str",
              help_message="Penalty for edits, on probability scale between 0 and 1."
                           + " Type a benchmark name to use pre-optimized threshold."),
    Parameter("out_file", "-f", "str",
              help_message="Name of the file where predicted sequences get stored."),
    Parameter("verbose", "-v", "boolean",
              help_message="Whether to print intermediate results."),
    Parameter("initialize", "-init", "boolean")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.corrector.beam_search.beam_search_corrector import BeamSearchCorrector
from src.interactive.sequence_generator import interactive_sequence_generator
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.helper.data_types import is_float
from src.corrector.beam_search.penalty_holder import PenaltyHolder
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import timestamp, time_diff


if __name__ == "__main__":
    model = UnidirectionalLMEstimator()
    model.load(parameters["model_name"])

    penalty = parameters["penalty"]
    if is_float(penalty):
        penalty = float(penalty)
    else:
        penalty_holder = PenaltyHolder()
        penalty = penalty_holder.get(parameters["model_name"], penalty)

    corrector = BeamSearchCorrector(model,
                                    backward=parameters["backward"],
                                    n_beams=parameters["n_beams"],
                                    average_log_likelihood=parameters["average_log_likelihood"],
                                    penalty=penalty)

    if parameters["benchmark"] == "0":
        benchmark = None
        sequences = interactive_sequence_generator()
    else:
        benchmark_name = parameters["benchmark"]
        benchmark_subset = Subset.TEST if benchmark_name in ("project", "doval") else Subset.DEVELOPMENT
        benchmark = Benchmark(benchmark_name,
                              subset=benchmark_subset)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)

    if parameters["benchmark"] == "0" or parameters["out_file"] == "0":
        file_writer = None
    else:
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + parameters["out_file"])
        if not parameters["initialize"]:
            file_writer.load()

    total_runtime = 0
    n_sequences = 0

    for s_i, sequence in enumerate(sequences):
        if s_i == parameters["n_sequences"] > 0:
            break
        if file_writer is not None and s_i < file_writer.n_sequences():
            continue
        print("sequence %i" % (s_i + 1))
        n_sequences += 1
        start_time = timestamp()
        predicted = corrector.correct(sequence, verbose=parameters["verbose"])
        runtime = time_diff(start_time)
        total_runtime += runtime

        if file_writer is not None:
            file_writer.add(predicted, runtime)
            file_writer.save()

        if parameters["verbose"]:
            print(predicted)

    print("total runtime = %.4f sec" % total_runtime)
    print("average runtime = %.4f sec" % (total_runtime / n_sequences))
