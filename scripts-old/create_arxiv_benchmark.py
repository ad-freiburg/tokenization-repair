import sys
import random

from project import src
from src.helper.files import read_lines, write_lines
from src.settings import paths
from src.arxiv.dataset import match_lines, to_input_file


if __name__ == "__main__":
    test = "test" in sys.argv
    random.seed(20201026)

    files_file = paths.ARXIV_TEST_FILES if test else paths.ARXIV_DEVELOPMENT_FILES
    subset_name = "test" if test else "development"

    files = read_lines(files_file)
    pairs = []
    for file in files:
        true_path = paths.ARXIV_GROUND_TRUTH_DIR + file
        input_path = paths.PDF_EXTRACT_DIR + to_input_file(file)
        matched = match_lines(true_path, input_path)
        pairs.extend(matched)

    random.shuffle(pairs)

    path = paths.BENCHMARKS_DIR + "arxiv/" + subset_name + "/"
    correct_sequences = [correct for _, correct in pairs]
    corrupt_sequences = [corrupt for corrupt, _ in pairs]
    write_lines(path + "correct.txt", correct_sequences)
    write_lines(path + "corrupt.txt", corrupt_sequences)
