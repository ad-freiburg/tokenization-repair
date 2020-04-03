import sys

import project
from src.settings import paths
from src.helper.files import read_lines


if __name__ == "__main__":
    file_name = sys.argv[1]

    for i in range(1, 11):
        p = i / 10
        benchmark_name = "random1" if p == 1 else ("random%.1f" % p)
        path = paths.RESULTS_DIR + benchmark_name + "/development/" + file_name
        mean_runtime = float(read_lines(path)[-1]) / 10000
        print("%s = %.2f" % (benchmark_name, mean_runtime))
