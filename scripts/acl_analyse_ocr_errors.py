import sys

import project
from src.helper.files import read_lines
from src.spelling.evaluation import get_ground_truth_labels, TokenErrorType
from src.plot.histogram import plot_rate_histogram, plot_histogram, save_histogram_data
from spelling_evaluation_space_preference import get_token_edit_labels

if __name__ == "__main__":
    folder = "/home/hertel/tokenization-repair-dumps/data/spelling/ACL/development/"
    out_folder = "acl_error_distribution_new/"
    n = 477

    analysis_type = sys.argv[1]  # "total", "spelling", "tokenization"
    exclude_zero = "no-zero" in sys.argv

    if analysis_type == "total":
        error_types = {TokenErrorType.TOKENIZATION_ERROR, TokenErrorType.OCR_ERROR, TokenErrorType.MIXED}
        error_name_label = "Total error"
    elif analysis_type == "spelling":
        error_types = {TokenErrorType.OCR_ERROR, TokenErrorType.MIXED}
        error_name_label = "Spelling error"
    else:
        error_types = {TokenErrorType.TOKENIZATION_ERROR}
        error_name_label = "Tokenization error"

    absolute_values = []
    error_rates = []

    for s_i, (correct, corrupt) in enumerate(zip(read_lines(folder + "spelling.txt"),
                                                 read_lines(folder + "corrupt.txt"))):
        token_errors = get_token_edit_labels(correct, corrupt)
        n_tokens = len(token_errors)
        if n_tokens < 30:  # or n_tokens > 40
            continue
        n_spelling_errors = len([error for error in token_errors if error in error_types])
        error_rate = n_spelling_errors / n_tokens
        print(n_spelling_errors, n_tokens, error_rate)
        absolute_values.append(n_spelling_errors)
        error_rates.append(error_rate)
        if s_i + 1 == n:
            break

    subtitle = "ACL development set (%i sequences)" % n

    save_histogram_data(error_rates, out_folder + f"{analysis_type}_error_rates.txt")
    plot_rate_histogram(error_rates,
                        title=f"{error_name_label} rates",
                        subtitle=subtitle,
                        xlabel=f"{error_name_label} rate (misspelled words / all words)",
                        save_path=out_folder + f"histogram_{analysis_type}_error_rates.png",
                        exclude_zero=exclude_zero)

    save_histogram_data(absolute_values, out_folder + f"{analysis_type}_errors.txt")
    plot_histogram(absolute_values,
                   title=f"{error_name_label}s",
                   subtitle=subtitle,
                   xlabel=f"{error_name_label}s (misspelled words / sequence)",
                   save_path=out_folder + f"histogram_{analysis_type}_errors.png",
                   exclude_zero=exclude_zero)
