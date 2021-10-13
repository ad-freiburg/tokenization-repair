import sys

from project import src
from src.evaluation.samples import get_space_corruptions
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.plot.histogram import plot_rate_histogram, plot_histogram, save_histogram_data


if __name__ == "__main__":
    exclude_zero = "no-zero" in sys.argv

    out_folder = "acl_error_distribution/"

    absolute_values = []
    error_rates = []

    for subset in (Subset.DEVELOPMENT, Subset.TEST):
        benchmark = Benchmark("ACL", subset)
        for correct, corrupt in benchmark.get_sequence_pairs(BenchmarkFiles.CORRUPT):
            print(corrupt)
            print(correct)
            edits = get_space_corruptions(correct, corrupt)
            n_edits = len(edits)
            n_chars = len(correct)
            ratio = n_edits / n_chars
            absolute_values.append(n_edits)
            error_rates.append(ratio)

    save_histogram_data(error_rates, out_folder + "tokenization_character_error_rates.txt")
    plot_rate_histogram(error_rates, title="Tokenization character error rates", subtitle="ACL development+test",
                        xlabel="Tokenization character error rate (whitespace errors / characters)",
                        save_path=out_folder + "histogram_tokenization_character_error_rates.png",
                        exclude_zero=exclude_zero)

    save_histogram_data(absolute_values, out_folder + "tokenization_character_errors.txt")
    plot_histogram(absolute_values, title="Tokenization character errors", subtitle="ACL development+test",
                   xlabel="Tokenization character errors (whitespace errors / sequence)",
                   save_path=out_folder + "histogram_tokenization_character_errors.png",
                   exclude_zero=exclude_zero)
