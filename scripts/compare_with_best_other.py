import project
from src.interactive.parameters import ParameterGetter, Parameter


params = [Parameter("approach", "-a", "str"),
          Parameter("noise_level", "-n", "float")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import numpy as np

from src.evaluation.results_holder import ResultsHolder, Metric
from src.benchmark.benchmark import get_error_probabilities, get_benchmark_name, Subset


if __name__ == "__main__":
    noise_level = parameters["noise_level"]
    approach= parameters["approach"]

    holder = ResultsHolder()

    for metric in (Metric.F1, Metric.SEQUENCE_ACCURACY):
        approach_vals = []
        best_other_vals = []
        for p in get_error_probabilities():
            values = holder.results[get_benchmark_name(noise_level, p)][Subset.TEST]
            approach_vals.append(values[approach][metric])
            best_other_vals.append(max(values[other][metric] for other in values if other != approach))
        print(metric)
        approach_mean = np.mean(approach_vals)
        other_mean = np.mean(best_other_vals)
        print(approach_mean, approach_vals)
        print(other_mean, best_other_vals)
        print("diff = %.4f" % (approach_mean - other_mean))