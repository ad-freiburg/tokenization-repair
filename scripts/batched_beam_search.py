import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("model", "-m", "str"),
          Parameter("benchmark", "-b", "str"),
          Parameter("subset", "-set", "str"),
          Parameter("sequences", "-seq", "str"),
          Parameter("n_sequences", "-n", "int"),
          Parameter("continue", "-c", "boolean"),
          Parameter("beams", "-w", "int"),
          Parameter("penalties", "-p", "str"),
          Parameter("penalty_multiplier", "-pm", "float"),
          Parameter("out_file", "-f", "str"),
          Parameter("labeling_model", "-labeling", "str")]
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
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator


if __name__ == "__main__":
    model_name = parameters["model"]

    model = UnidirectionalModel(model_name)
    backward = model.model.specification.backward

    if parameters["labeling_model"] == "0":
        labeling_model = None
    else:
        labeling_model = BidirectionalLabelingEstimator()
        labeling_model.load("labeling")

    benchmark_name = parameters["benchmark"]

    if benchmark_name == "0":
        sequences = interactive_sequence_generator()
        file_writer = None
    else:
        benchmark = Benchmark(benchmark_name, get_subset(parameters["subset"]))
        if parameters["sequences"] == "corrupt":
            sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        else:
            sequences = benchmark.get_predicted_sequences(parameters["sequences"])
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + parameters["out_file"])
        if parameters["continue"]:
            file_writer.load()

    penalties = parameters["penalties"]
    if penalties == "0":
        insertion_penalty = deletion_penalty = 0
    else:
        penalty_holder = PenaltyHolder(two_pass=parameters["sequences"] != "corrupt")
        penalty_name = model_name
        if parameters["labeling_model"] != "0":
            penalty_name += "_" + parameters["labeling_model"]
        insertion_penalty, deletion_penalty = penalty_holder.get(penalty_name, penalties)
        if isinstance(parameters["penalty_multiplier"], list):
            insertion_penalty_multiplier, deletion_penalty_multiplier = parameters["penalty_multiplier"]
        else:
            insertion_penalty_multiplier = deletion_penalty_multiplier = parameters["penalty_multiplier"]
        insertion_penalty = insertion_penalty_multiplier * insertion_penalty
        deletion_penalty = deletion_penalty_multiplier * deletion_penalty

    corrector = BatchedBeamSearchCorrector(model.model,
                                           insertion_penalty=insertion_penalty,
                                           deletion_penalty=deletion_penalty,
                                           n_beams=parameters["beams"],
                                           verbose=benchmark_name == "0",
                                           labeling_model=labeling_model)

    for s_i, sequence in enumerate(sequences):
        if s_i == parameters["n_sequences"]:
            break
        elif file_writer is not None and s_i < file_writer.n_sequences():
            continue
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        if file_writer is not None:
            file_writer.add(predicted, runtime)

        if file_writer is not None:
            if (s_i + 1) % 100 == 0:
                file_writer.save()
