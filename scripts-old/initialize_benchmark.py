import sys

from project import src
from src.benchmark.extended_benchmark import ExtendedBenchmark
from src.benchmark.subset import Subset


if __name__ == "__main__":
    benchmark_name = sys.argv[1]
    subset = Subset.DEVELOPMENT if benchmark_name != "project" else Subset.TEST
    prediction_file_names = sys.argv[2:] if len(sys.argv) > 2 else None
    benchmark = ExtendedBenchmark("random",
                                  Subset.DEVELOPMENT,
                                  subdir="1000",
                                  prediction_file_names=prediction_file_names)
    benchmark.initialize()
    benchmark.register_predictions()
    benchmark.write_csv()
