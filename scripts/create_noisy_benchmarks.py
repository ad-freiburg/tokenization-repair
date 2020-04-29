from typing import List, Tuple

import numpy as np

import project
from src.noise.typo_noise_inducer import TypoNoiseInducer
from src.sequence.token_corruptor import TokenCorruptor
from src.settings import constants
from src.helper.files import write_lines, make_directory_recursive
from src.settings import paths


SEED = 3010
NOISE_LEVELS = [0, 0.1]
ERROR_PROBABILITIES = [0.1, 1, np.inf]


def insert_noise(sequences: List[str],
                 noise_level: float) -> List[str]:
    if noise_level > 0:
        inducer = TypoNoiseInducer(p=noise_level,
                                   seed=SEED)
        sequences = [inducer.induce_noise(sequence) for sequence in sequences]
    return sequences


def corrupt_tokenization(sequences: List[str],
                         p: float) -> List[str]:
    if p == np.inf:
        sequences = [sequence.replace(' ', '') for sequence in sequences]
    else:
        corruptor = TokenCorruptor(p=p,
                                   positions_per_token=constants.POSITIONS_PER_TOKEN,
                                   token_pairs_per_token=constants.TOKEN_PAIRS_PER_TOKEN,
                                   seed=SEED)
        sequences = [corruptor.corrupt(sequence) for sequence in sequences]
    return sequences


def benchmark_directories(noise_level: float,
                          p: float):
    noise_level_str = ("%.1f" % noise_level) if noise_level > 0 else "0"
    if p == np.inf:
        p_str = "inf"
    elif p == 1:
        p_str = "1"
    else:
        p_str = "%.1f" % p
    base_path = paths.BENCHMARKS_DIR + "%s_%s/" % (noise_level_str, p_str)
    tuning_path = base_path + "tuning/"
    development_path = base_path + "development/"
    test_path = base_path + "test/"
    return tuning_path, development_path, test_path


def file_paths(dir_path: str):
    correct_path = dir_path + "correct.txt"
    corrupt_path = dir_path + "corrupt.txt"
    return correct_path, corrupt_path


if __name__ == "__main__":
    from src.helper.files import read_lines

    tuning_sequences = read_lines(paths.WIKI_TUNING_SENTENCES)
    development_sequences = read_lines(paths.WIKI_DEVELOPMENT_SENTENCES)
    test_sequences = read_lines(paths.WIKI_TEST_SENTENCES)

    for noise_level in NOISE_LEVELS:
        tuning_ground_truth_sequences = insert_noise(tuning_sequences, noise_level)
        development_ground_truth_sequences = insert_noise(development_sequences, noise_level)
        test_ground_truth_sequnences = insert_noise(test_sequences, noise_level)

        for p in ERROR_PROBABILITIES:
            print(noise_level, p)
            tuning_corrupt_sequences = corrupt_tokenization(tuning_ground_truth_sequences, p)
            development_corrupt_sequences = corrupt_tokenization(development_ground_truth_sequences, p)
            test_corrupt_sequences = corrupt_tokenization(test_ground_truth_sequnences, p)

            tuning_path, development_path, test_path = benchmark_directories(noise_level, p)
            make_directory_recursive(tuning_path)
            make_directory_recursive(development_path)
            make_directory_recursive(test_path)
            tune_correct_path, tune_corrupt_path = file_paths(tuning_path)
            dev_correct_path, dev_corrupt_path = file_paths(development_path)
            test_correct_path, test_corrupt_path = file_paths(test_path)

            write_lines(tune_correct_path, tuning_ground_truth_sequences)
            write_lines(tune_corrupt_path, tuning_corrupt_sequences)
            write_lines(dev_correct_path, development_ground_truth_sequences)
            write_lines(dev_corrupt_path, development_corrupt_sequences)
            write_lines(test_correct_path, test_ground_truth_sequnences)
            write_lines(test_corrupt_path, test_corrupt_sequences)
