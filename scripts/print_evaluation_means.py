import numpy as np

import project
from src.benchmark.benchmark import get_benchmark_name, get_error_probabilities, Subset
from src.evaluation.results_holder import ResultsHolder, Metric


NOISE_LEVELS = [0, 0.1, 0.2]
APPROACHES = ["combined",
              "combined_robust",
              "softmax",
              "softmax_robust",
              "sigmoid",
              "sigmoid_robust",
              "beam_search",
              "beam_search_robust",
              "do_nothing"]
NAME_LEN = 21
METRICS = [Metric.F1, Metric.SEQUENCE_ACCURACY]


if __name__ == "__main__":
    holder = ResultsHolder()

    for approach in APPROACHES:
        print_str = "  "
        print_str += approach.replace('_', ' ')
        print_str += ' ' * (NAME_LEN - len(approach))
        for noise_level in NOISE_LEVELS:
            for metric in METRICS:
                values = [holder.get(get_benchmark_name(noise_level, p), Subset.TEST, approach, metric)
                          for p in get_error_probabilities()]
                mean = float(np.mean(values))
                print_str += "& %.2f\\,\\%% " % (mean * 100)
        print_str += "\\\\"
        print(print_str)