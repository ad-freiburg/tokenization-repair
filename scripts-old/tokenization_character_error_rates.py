import sys
import matplotlib.pyplot as plt

import project
from src.helper.files import read_lines
from src.evaluation.evaluator import get_space_corruptions


if __name__ == "__main__":
    correct_file = sys.argv[1]
    corrupt_file = sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else None

    error_rates = []

    for correct, corrupt in zip(read_lines(correct_file), read_lines(corrupt_file)):
        n_chars = len(correct)
        errors = get_space_corruptions(correct, corrupt)
        n_errors = len(errors)
        error_rate = n_errors / n_chars
        print(error_rate, n_errors, n_chars)
        error_rates.append(error_rate)

    if out_file is not None:
        with open(out_file, "w") as f:
            for error_rate in error_rates:
                f.write(str(error_rate) + "\n")

    plt.hist(error_rates, bins=[-0.01, 1e-16] + [i / 100 for i in range(1, 101)])
    plt.show()
