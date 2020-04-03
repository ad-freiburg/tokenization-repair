import project

from src.evaluation.results_holder import ResultsHolder, Metric
from src.benchmark.benchmark import Subset


if __name__ == "__main__":
    holder = ResultsHolder()
    benchmarks = [key for key in holder.results]
    for benchmark in benchmarks:
        for approach in ("combined", "combined_robust"):
            values = holder.results[benchmark][Subset.TEST][approach]
            new_name = approach + "_old"
            holder.results[benchmark][Subset.TEST][new_name] = values
    holder.save()
