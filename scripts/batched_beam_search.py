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
          Parameter("penalty_modifier", "-pm", "str"),
          Parameter("out_file", "-f", "str"),
          Parameter("labeling_model", "-labeling", "str"),
          Parameter("lookahead", "-l", "int")]
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


def modify_penalties(insertion_penalty: float, deletion_penalty: float):
    modifiers = parameters["penalty_modifier"]
    if not isinstance(modifiers, list):
        modifiers = [modifiers, modifiers]
    penalties = [insertion_penalty, deletion_penalty]
    for i, modifier in enumerate(modifiers):
        if modifier[0] in ('+', '-', '*'):
            operator = modifier[0]
            value = float(modifier[1:])
            if operator == '+':
                penalties[i] -= value
            elif operator == '-':
                penalties[i] += value
            else:
                penalties[i] *= value
    return penalties


if __name__ == "__main__":
    model_name = parameters["model"]

    model = UnidirectionalModel(model_name)
    backward = model.model.specification.backward

    if parameters["labeling_model"] == "0":
        labeling_model = None
    else:
        labeling_model = BidirectionalLabelingEstimator()
        labeling_model.load(parameters["labeling_model"])

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
        # penalty_holder = PenaltyHolder(two_pass=parameters["sequences"] != "corrupt")
        penalty_holder = PenaltyHolder(seq_acc=True)  # TODO
        penalty_name = model_name
        if parameters["labeling_model"] != "0":
            penalty_name += "_" + parameters["labeling_model"]
        penalty_name = penalty_name.replace("_acl", "")  # dirty hack for fine-tuned models which have no fitted penalties
        #if parameters["lookahead"] > 0:
        #    penalty_name += "_lookahead%i" % parameters["lookahead"]  # TODO
        insertion_penalty, deletion_penalty = penalty_holder.get(penalty_name, penalties)
    insertion_penalty, deletion_penalty = modify_penalties(insertion_penalty, deletion_penalty)
    print("penalties:", insertion_penalty, deletion_penalty)

    #add_epsilon = benchmark.name == "nastase" and model_name == "arxiv_fwd1024" and \
    #              parameters["labeling_model"] == "arxiv_labeling"
    add_epsilon = parameters["labeling_model"] != "0"

    corrector = BatchedBeamSearchCorrector(model.model,
                                           insertion_penalty=insertion_penalty,
                                           deletion_penalty=deletion_penalty,
                                           n_beams=parameters["beams"],
                                           verbose=benchmark_name == "0",
                                           labeling_model=labeling_model,
                                           add_epsilon=add_epsilon)

    for s_i, sequence in enumerate(sequences):
        if s_i == parameters["n_sequences"]:
            break
        elif file_writer is not None and s_i < file_writer.n_sequences():
            continue
        start_time = timestamp()
        if benchmark_name == "acl_all" and sum(1 if c == "?" else 0 for c in sequence) > 3:  # ignore too many ?s
            predicted = sequence
        else:
            predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        if file_writer is not None:
            file_writer.add(predicted, runtime)

        if file_writer is not None:
            if (s_i + 1) % 100 == 0:
                file_writer.save()

    if file_writer is not None:
        file_writer.save()
