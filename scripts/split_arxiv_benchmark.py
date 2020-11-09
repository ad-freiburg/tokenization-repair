import random

from project import src
from src.arxiv.dataset import get_files, match_lines, to_input_file
from src.settings import paths
from src.helper.files import write_lines


if __name__ == "__main__":
    random.seed(20201026)

    files = get_files()

    matched_files = []
    unmatched_files = []

    for file in files:
        truth_file = paths.ARXIV_GROUND_TRUTH_DIR + file
        input_file = paths.PDF_EXTRACT_DIR + to_input_file(file)
        matched = match_lines(truth_file, input_file)
        print(truth_file, input_file, len(matched))
        if len(matched) > 0:
            matched_files.append(file)
        else:
            unmatched_files.append(file)

    print("%i matched" % len(matched_files))
    print("%i unmatched" % len(unmatched_files))

    random.shuffle(matched_files)

    write_lines(paths.ARXIV_DEVELOPMENT_FILES, sorted(matched_files[:1000]))
    write_lines(paths.ARXIV_TEST_FILES, sorted(matched_files[1000:2000]))

    training_files = sorted(matched_files[2000:] + unmatched_files)
    write_lines(paths.ARXIV_TRAINING_FILES, training_files)
