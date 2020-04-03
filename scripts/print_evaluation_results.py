import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("noise_level", "-n", "float"),
          Parameter("approach", "-a", "str")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import numpy as np

from src.evaluation.results_holder import ResultsHolder, Metric
from src.benchmark.benchmark import get_benchmark_name, get_error_probabilities, Subset


if __name__ == "__main__":
    approach = parameters["approach"]
    holder = ResultsHolder()
    metrics = [Metric.F1, Metric.SEQUENCE_ACCURACY, Metric.MEAN_RUNTIME]
    values = {metric: [] for metric in metrics}
    for p in get_error_probabilities():
        benchmark_name = get_benchmark_name(parameters["noise_level"], p)
        benchmark_values = []
        for metric in metrics:
            value = holder.get(benchmark_name, Subset.TEST, approach, metric)
            benchmark_values.append(value)
            values[metric].append(value)
        print_name = benchmark_name[:7]
        print_name += ' ' * (7 - len(print_name))
        print(print_name, ' '.join(str(value) for value in benchmark_values))
    for metric in metrics:
        metric_values = values[metric]
        print(metric, "mean = %.4f (min = %.4f, max = %.4f)" % (np.mean(metric_values),
                                                                min(metric_values),
                                                                max(metric_values)))
