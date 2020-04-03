"""Fits thresholds for the greedy tokenization repair approach."""

from project import src
from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("model_type", "-type", "str",
              help_message="Choose 'bidir' or 'combined'.",
              dependencies=[
                  ("bidir",
                   [Parameter("model_name", "-name", "str",
                              help_message="Name of the model.")]),
                  ("combined",
                   [Parameter("fwd_model_name", "-fwd", "str",
                              help_message="Name of the forward model."),
                    Parameter("bwd_model_name", "-bwd", "str",
                              help_message="Name of the backward model.")
                    ])
              ]),
    Parameter("benchmark", "-benchmark", "str"),
    Parameter("noise", "-noise", "str"),
    Parameter("insert", "-insert", "boolean"),
    Parameter("initialize", "-init", "boolean"),
    Parameter("threshold", "-t", "float")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.load.load_char_lm import load_char_lm
from src.corrector.greedy_corrector import GreedyCorrector
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.corrector.threshold_fitter import ThresholdFitter
from src.corrector.threshold_fitter_holder import ThresholdFitterHolder
from src.corrector.threshold_holder import ThresholdHolder, ThresholdType
from src.helper.time import timestamp, time_diff


if __name__ == "__main__":
    model = load_char_lm(model_type=parameters["model_type"],
                         model_name=parameters["model_name"] if "model_name" in parameters
                         else parameters["fwd_model_name"],
                         bwd_model_name=parameters["bwd_model_name"] if "bwd_model_name" in parameters else None)

    if parameters["threshold"] <= 0:
        if parameters["insert"]:
            if parameters["model_type"] == "combined":
                THRESHOLD = 0.99
            elif "softmax" in parameters["model_name"]:
                THRESHOLD = 0.95
            else:
                THRESHOLD = 0.1
        else:
            THRESHOLD = 0.01
    else:
        THRESHOLD = parameters["threshold"]

    corrupt_file = BenchmarkFiles.DELETIONS if parameters["insert"] else BenchmarkFiles.INSERTIONS

    benchmark = Benchmark(parameters["benchmark"],
                          Subset.DEVELOPMENT)

    corrector = GreedyCorrector(model,
                                insert=parameters["insert"],
                                delete=not parameters["insert"],
                                insertion_threshold=THRESHOLD,
                                deletion_threshold=THRESHOLD)

    fitter_holder = ThresholdFitterHolder(model_name=parameters["model_name"] if "model_name" in parameters else None,
                                          fwd_model_name=parameters["fwd_model_name"]
                                          if "fwd_model_name" in parameters else None,
                                          bwd_model_name=parameters["bwd_model_name"]
                                          if "bwd_model_name" in parameters else None,
                                          benchmark_name=parameters["benchmark"],
                                          insert=parameters["insert"])

    if parameters["initialize"]:
        fitter = ThresholdFitter()
    else:
        fitter = fitter_holder.load()

    for s_i, (correct, corrupt) in enumerate(benchmark.get_sequence_pairs(corrupt_file)):
        if s_i < fitter.n_sequences:
            continue
        print("sequence %i" % s_i)
        print(correct)
        print(corrupt)
        start_time = timestamp()
        predicted = corrector.correct(corrupt, return_details=True)
        predicted.set_runtime(time_diff(start_time))
        fitter.add_example(correct=correct, corrupt=corrupt, predicted=predicted)
        fitter_holder.dump(fitter)
        print()

    fitter.fit()

    # plot
    if parameters["model_type"] == "bidir":
        title = parameters["model_name"]
    else:
        title = "combined %s %s" % (parameters["fwd_model_name"], parameters["bwd_model_name"])
    if parameters["insert"]:
        title += " insert"
    else:
        title += " delete"
    fitter.plot(title, save=True)

    # best threshold
    fitter.print_best()

    # save threshold
    if parameters["model_type"] == "bidir":
        model_name = parameters["model_name"]
        fwd_model_name = None
        bwd_model_name = None
    else:
        fwd_model_name = parameters["fwd_model_name"]
        bwd_model_name = parameters["bwd_model_name"]
        model_name = None

    holder = ThresholdHolder()
    threshold_type = ThresholdType.INSERTION_THRESHOLD if parameters["insert"] else ThresholdType.DELETION_THRESHOLD
    holder.set_threshold(threshold_type=threshold_type,
                         model_name=model_name,
                         fwd_model_name=fwd_model_name,
                         bwd_model_name=bwd_model_name,
                         threshold=fitter.get_threshold(),
                         noise_type=parameters["noise"])
