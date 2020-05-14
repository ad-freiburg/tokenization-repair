import project
from src.interactive.parameters import Parameter, ParameterGetter

params = [Parameter("model", "-m", "str"),
          Parameter("benchmark", "-b", "str"),
          Parameter("subset", "-s", "str"),
          Parameter("penalties", "-p", "str"),
          Parameter("out_file", "-f", "str")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()

from src.models.char_lm.unidirectional_model import UnidirectionalModel
from src.benchmark.benchmark import Benchmark, BenchmarkFiles, get_subset
from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
from src.corrector.beam_search.penalty_holder import PenaltyHolder
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp
from src.interactive.sequence_generator import interactive_sequence_generator


if __name__ == "__main__":
    model_name = parameters["model"]

    model = UnidirectionalModel(model_name)
    backward = model.model.specification.backward

    benchmark_name = parameters["benchmark"]

    if benchmark_name == "0":
        sequences = interactive_sequence_generator()
        file_writer = None
    else:
        benchmark = Benchmark(benchmark_name, get_subset(parameters["subset"]))
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + parameters["out_file"])

    penalties = parameters["penalties"]
    if penalties == "0":
        insertion_penalty = deletion_penalty = 0
    else:
        penalty_holder = PenaltyHolder()
        insertion_penalty, deletion_penalty = penalty_holder.get(model_name, penalties)

    corrector = BatchedBeamSearchCorrector(model.model,
                                           insertion_penalty=insertion_penalty,
                                           deletion_penalty=deletion_penalty,
                                           n_beams=5,
                                           verbose=benchmark_name == "0")

    for s_i, sequence in enumerate(sequences):
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        if file_writer is not None:
            file_writer.add(predicted, runtime)

    if file_writer is not None:
        file_writer.save()
