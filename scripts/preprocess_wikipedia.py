"""
Preprocesses the Wikipedia raw data into readable formats.
Supported output formats are paragraph files or single-file.
"""

from typing import Tuple, List

import sys
import random

from project import src
from src.helper.files import path_exists, write_file, make_directory, get_files, remove_file, read_sequences
from src.helper.pickle import load_object, dump_object
from src.sequence.token_corruptor import TokenCorruptor
from src.data.wiki.raw_wikipedia import split_dataset, count_paragraphs, detect_tables, write_clean_articles, \
    write_dataset_split_files
from src.data.wikipedia import Wikipedia, Sequence
from src.settings import paths, constants


SPLITS = ["training", "development", "test"]


def find_wikipedia_dir() -> str:
    """Searches for the extracted Wikipedia files at pre-defined paths.

    :return: Absolute path to the extracted Wikipedia files.
    """
    dirs = ["/mnt/4E83539B6CE67342/tokenization-repair-dumps/data/wikipedia/",
            "/local/data/hertelm/wikipedia/",
            "/local/hdd/exports/data/matthias-hertel/tokenization-repair-dumps/data/wikipedia/",
            "/project/master/hertelm/tokenization-repair-dumps/data/wikipedia/"]
    for dir in dirs:
        if path_exists(dir):
            return dir
    raise Exception("Could not find extracted Wikipedia directory.")


def get_article_ids_split(out_directory: str) -> Tuple[List[int], List[int], List[int]]:
    """Reads the article IDs of the training, development and test partitions from disk.

    :param out_directory: directory where the IDs are stored
    :return: lists of training, development and test article IDs
    """
    training_ids_path = out_directory + "training_article_ids.pkl"
    development_ids_path = out_directory + "development_article_ids.pkl"
    test_ids_path = out_directory + "test_article_ids.pkl"
    training_ids = load_object(training_ids_path)
    development_ids = load_object(development_ids_path)
    test_ids = load_object(test_ids_path)
    return training_ids, development_ids, test_ids


def count_tokens_and_insert_positions(split: str):
    """Counts and prints the number of tokens and number of insertion positions in the given partition.

    Insertion positions are positions not preceded or followed by a space. Assumes the sequences not to have leading
    or trailing spaces.
    Runs through the dataset in paragraph format.

    :param split: partition name, either training, development or test
    """
    n_sequences = 0
    tokens_total = 0
    insertion_positions_total = 0
    for file in Wikipedia.file_iterator(benchmark_name="correct", split=split):
        sequence = Wikipedia.get_sequence(file)
        n_spaces = sequence.count(' ')
        n_tokens = n_spaces + 1
        n_insert_positions = len(sequence) - 2 * n_spaces - 1
        tokens_total += n_tokens
        n_sequences += 1
        insertion_positions_total += n_insert_positions
    print("%i insertion positions in %i tokens (%i sequences)." %
          (insertion_positions_total, tokens_total, n_sequences))


def _benchmark_name(p):
    """Creates the name for a benchmark with the given corruption probability p.

    The name is corrupt_spaces_p_X.XXXXXXXX without trailing zeros.

    :param p: corruption probability
    :return: name of the benchmark
    """
    p_str = "%.8f" % p
    while p_str[-1] == '0':
        p_str = p_str[:-1]
    benchmark_name = "corrupt_spaces_p_%s" % p_str
    return benchmark_name


def _corruptor(p: float,
               seed: int
               ) -> TokenCorruptor:
    """Token corruptor with the given p and seed.

    Uses the number of insertion positions and token pairs defined in src.settings.constants.

    :param p: corruption probability
    :param seed: random seed
    :return: corruptor
    """
    corruptor = TokenCorruptor(p=p,
                               positions_per_token=constants.POSITIONS_PER_TOKEN,
                               token_pairs_per_token=constants.TOKEN_PAIRS_PER_TOKEN,
                               seed=seed)
    return corruptor


def corrupt_dataset(directory: str,
                    p: float,
                    splits: List[str],
                    seed: int):
    """Generates a corrupt dataset in paragraph format.

    The format is as follows:
    directory
    ---| training
    -------| <benchmark_name>
    -----------| texts
    ---------------| 0000
    -------------------| <sequence_file_name>
    -------------------| ...
    ---------------| ...
    ---| development
    ---| test

    :param directory: output directory
    :param p: corruption probability
    :param splits: subset of {training, development, test}, provided as a list
    :param seed: corruption random seed
    """
    corruptor = _corruptor(p, seed)
    benchmark_name = _benchmark_name(p)
    for split in splits:
        benchmark_split_dir = directory + split + "/" + benchmark_name + "/"
        if not path_exists(benchmark_split_dir):
            make_directory(benchmark_split_dir)
        text_dir = benchmark_split_dir + "texts/"
        if not path_exists(text_dir):
            make_directory(text_dir)
        for file in Wikipedia.file_iterator(benchmark_name="correct", split=split):
            sequence = Wikipedia.get_sequence(file)
            corrupt = corruptor.corrupt(sequence)
            path_split = file.split('/')
            path_split[-4] = benchmark_name
            folder = paths.WIKI_DIR + '/'.join(path_split[:-1])
            if not path_exists(folder):
                make_directory(folder)
            path = paths.WIKI_DIR + '/'.join(path_split)
            write_file(path, corrupt)
    corruptor.print_summary()


def corrupt_dataset_single(p, splits, seed):
    """Creates a corrupt dataset in single-file format.

    :param p: corruption probability
    :param splits: subset of {training, development, test}, provided as a list
    :param seed: corruption random seed
    """
    corruptor = _corruptor(p, seed)
    benchmark_name = _benchmark_name(p)
    for split in splits:
        if split == "training":
            correct_sequences_path = paths.WIKI_TRAINING_SEQUENCES
        elif split == "development":
            correct_sequences_path = paths.WIKI_DEVELOPMENT_SEQUENCES
        else:
            correct_sequences_path = paths.WIKI_TEST_SEQUENCES
        correct_sequences = load_object(correct_sequences_path)
        corrupt_sequences = []
        byte_position = 0
        in_path = paths.WIKI_SINGLE_DIR + split + ".txt"
        out_path = paths.WIKI_SINGLE_DIR + "%s_%s.txt" % (benchmark_name, split)
        with open(out_path, 'wb') as out_file:
            for s_i, sequence in enumerate(read_sequences(in_path)):
                corrupt = corruptor.corrupt(sequence)
                s_id = correct_sequences[s_i].id
                bytes = (corrupt + '\n').encode("utf8")
                out_file.write(bytes)
                byte_len = out_file.tell() - byte_position
                char_len = len(corrupt)
                corrupt_sequences.append(Sequence(s_id, split, byte_position, byte_len, char_len))
                byte_position += byte_len
        corrupt_sequences_path = paths.WIKI_OUT_DIR + "%s_%s_sequences.pkl" % (benchmark_name, split)
        dump_object(corrupt_sequences, corrupt_sequences_path)
        corruptor.print_summary()


def _remove_training_split_files():
    for file in get_files(paths.WIKI_TRAINING_SPLIT_DIR):
        remove_file(paths.WIKI_TRAINING_SPLIT_DIR + file)


def split_training_set():
    """Splits the training data set in single-file format into files of sequences of equal length."""

    _remove_training_split_files()
    length_counts = {}
    with open(paths.WIKI_TRAINING_FILE) as training_file:
        while True:
            line = training_file.readline()
            if line == "":
                break
            sequence = line[:-1]
            seq_len = len(sequence)
            if seq_len not in length_counts:
                length_counts[seq_len] = 0
            path = paths.WIKI_TRAINING_SPLIT_DIR + "%i.txt" % seq_len
            with open(path, 'a', encoding="utf8") as file:
                file.write(sequence + "\n")
            length_counts[seq_len] += 1
    dump_object(length_counts, paths.WIKI_TRAINING_SEQUENCE_COUNTS)


if __name__ == "__main__":
    MODE = sys.argv[1]
    random.seed(42)

    wiki_base_dir = find_wikipedia_dir()
    wiki_text_directory = wiki_base_dir + "text"
    out_directory = wiki_base_dir

    if MODE == "split":
        split_articles = int(sys.argv[2])  # number of articles in development and test sets
        split_dataset(wiki_text_directory, out_directory, split_articles)

    if MODE == "paragraphs":
        count_paragraphs(wiki_text_directory)

    if MODE == "tables":
        detect_tables(wiki_text_directory)

    if MODE == "correct" or MODE == "correct-single":
        _, development_ids, test_ids = get_article_ids_split(out_directory)
        development_ids = set(development_ids)
        test_ids = set(test_ids)
        if MODE == "correct":
            write_clean_articles(wiki_text_directory, out_directory, development_ids, test_ids)
        else:
            write_dataset_split_files(wiki_text_directory, development_ids, test_ids)

    if MODE == "training":
        split_training_set()

    if MODE == "sequences":
        seed = int(sys.argv[2])
        n_sequences = int(sys.argv[3])
        for sequence in Wikipedia.training_sequences(n_sequences, seed):
            print(sequence)

    if MODE == "tokens":
        if len(sys.argv) > 2:
            split = sys.argv[2]
        else:
            split = "training"
        count_tokens_and_insert_positions(split)

    if MODE == "corrupt" or MODE == "corrupt-single":
        p = float(sys.argv[2])
        seed = int(sys.argv[3])
        splits = sys.argv[4:]
        if MODE == "corrupt":
            corrupt_dataset(out_directory, p, splits, seed)
        else:
            corrupt_dataset_single(p, splits, seed)

    if MODE == "batch_iterator":
        batch_size = int(sys.argv[2])
        batches = Wikipedia.training_batches(batch_size)

    if MODE == "benchmark":
        benchmark_name = sys.argv[2]
        split = sys.argv[3]
        seed = int(sys.argv[4])
        if seed < 0:
            seed = None
        n = int(sys.argv[5])
        for correct, corrupt in Wikipedia.benchmark_sequence_pairs(benchmark_name, split, seed=seed, n=n):
            print(correct)
            print(corrupt)
            print()
