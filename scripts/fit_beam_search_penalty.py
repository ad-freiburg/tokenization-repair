from project import src
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("model_name", "-name", "str"),
          Parameter("labeling_model_name", "-labeling", "str"),
          Parameter("noise_level", "-noise", "float"),
          Parameter("two_pass", "-tp", "str"),
          Parameter("n_sequences", "-n", "int")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import numpy as np

from src.benchmark.benchmark import get_benchmark, Subset
from src.benchmark.two_pass_benchmark import get_two_pass_benchmark
from src.corrector.beam_search.penalty_fitter import PenaltyFitter
from src.corrector.beam_search.penalty_holder import PenaltyHolder


if __name__ == "__main__":
    error_probabilities = [0.1, 1]
    if parameters["two_pass"] == "0":
        benchmarks = [get_benchmark(noise_level=parameters["noise_level"],
                                    p=p,
                                    subset=Subset.TUNING) for p in error_probabilities]
    else:
        error_probabilities.append(np.inf)
        benchmarks = [get_two_pass_benchmark(parameters["noise_level"], p, Subset.TUNING, parameters["two_pass"])
                      for p in error_probabilities]

    labeling_model_name = None if parameters["labeling_model_name"] == "0" else parameters["labeling_model_name"]
    fitter = PenaltyFitter(parameters["model_name"],
                           labeling_model_name=labeling_model_name,
                           n_sequences=parameters["n_sequences"])

    penalties = fitter.fit(benchmarks)
    holder = PenaltyHolder(two_pass=parameters["two_pass"] != "0", autosave=False)

    for benchmark_name in penalties:
        insertion_penalty, deletion_penalty = penalties[benchmark_name]
        save_name = parameters["model_name"]
        if parameters["labeling_model_name"] != "0":
            save_name += "_" + parameters["labeling_model_name"]
        holder.set(model_name=save_name,
                   benchmark_name=benchmark_name,
                   insertion_penalty=insertion_penalty,
                   deletion_penalty=deletion_penalty)

    holder.save()
