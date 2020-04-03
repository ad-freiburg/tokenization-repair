from project import src
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("model_name", "-name", "str"),
          Parameter("noise_level", "-noise", "float")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import numpy as np

from src.benchmark.benchmark import get_benchmark, Subset
from src.corrector.beam_search.penalty_fitter import PenaltyFitter
from src.corrector.beam_search.penalty_holder import PenaltyHolder


if __name__ == "__main__":
    error_probabilities = np.arange(0.1, 1.1, 0.1)
    benchmarks = [get_benchmark(noise_level=parameters["noise_level"],
                                p=p,
                                subset=Subset.DEVELOPMENT) for p in error_probabilities]
    fitter = PenaltyFitter(parameters["model_name"])
    penalties = fitter.fit(benchmarks)
    holder = PenaltyHolder(autosave=False)
    for benchmark_name in penalties:
        insertion_penalty, deletion_penalty = penalties[benchmark_name]
        holder.set(model_name=parameters["model_name"],
                   benchmark_name=benchmark_name,
                   insertion_penalty=insertion_penalty,
                   deletion_penalty=deletion_penalty)
    holder.save()
