import random
import json

from src.helper.files import get_files, read_lines, write_file
from src.helper.pickle import dump_object
from src.data.wiki.benchmark_file_name_generator import BenchmarkFileNameGenerator
from src.data.preprocessing import preprocess_sequence
from src.data.wikipedia import Sequence, DatasetSplit
from src.settings import paths


FILE_ENCODING = "utf8"


def get_files_depth_two(directory):
    """
    Returns the paths to all files from the directories in the given directory.

    :param directory: a directory with directories containing the wanted files
    :return: list of full paths to all files at depth two
    """
    subdirs = get_files(directory)
    files = []
    for subdir in sorted(subdirs):
        path = directory + "/" + subdir + "/"
        subdir_files = get_files(path)
        for file in sorted(subdir_files):
            files.append(path + file)
    return files


def get_article_jsons(wiki_text_directory):
    """
    Reads all articles as jsons from an extracted Wikipedia dump.

    :param wiki_text_directory: Link to a directory created by the WikiExtractor script.
        Assumes subdirectories to contain files where each line corresponds to an article json.
    :return: iterator over article jsons
    """
    json_files = get_files_depth_two(wiki_text_directory)
    for file in json_files:
        lines = read_lines(file)
        for line in lines:
            article = json.loads(line)
            yield article


def split_article_ids(wiki_text_directory, n_split):
    """
    Runs through the extracted jsons, collects all article IDs and splits them into training, development and test sets.

    :param wiki_text_directory: Link to a directory created by the WikiExtractor script.
        Assumes subdirectories to contain files where each line corresponds to an article json.
    :param n_split: number of articles in the development and test sets
    :return: three lists containing the article IDs of the training, development and test sets
    """
    article_ids = [article["id"] for article in get_article_jsons(wiki_text_directory)]
    assert(len(article_ids) > 2 * n_split)
    random.shuffle(article_ids)
    development_ids = article_ids[:n_split]
    test_ids = article_ids[n_split:(2 * n_split)]
    training_ids = article_ids[(2 * n_split):]
    return training_ids, development_ids, test_ids


def split_dataset(wiki_text_directory, out_directory, n_split):
    """
    Reads all article IDs from an extracted wikipedia dump, splits them into training, development and test sets and
    pickles the three sets as lists.
    :param wiki_text_directory: Link to a directory created by the WikiExtractor script.
        Assumes subdirectories to contain files where each line corresponds to an article json.
    :param out_directory: directory where the ID lists are stored
    :param n_split: number of articles of the training and test sets
    """
    training_ids, development_ids, test_ids = split_article_ids(wiki_text_directory, n_split)
    training_ids_path = out_directory + "training_article_ids.pkl"
    development_ids_path = out_directory + "development_article_ids.pkl"
    test_ids_path = out_directory + "test_article_ids.pkl"
    dump_object(training_ids, training_ids_path)
    dump_object(development_ids, development_ids_path)
    dump_object(test_ids, test_ids_path)
    print("Split dataset into %i training articles, %i development articles, %i test articles." %
          (len(training_ids), len(development_ids), len(test_ids)))


def get_paragraphs(article_json):
    """
    Retrieve paragraphs from an article.
    Paragraphs get preprocessed and filtered,

    :param article_json: the article as json string
    :return: list of paragraphs as strings
    """
    text = article_json["text"]
    lines = text.split("\n")
    lines = [preprocess_sequence(line) for line in lines]
    lines = [line for line in lines if len(line) > 0]
    lines = [line for line in lines if line != "<onlyinclude></onlyinclude>"]
    return lines


def write_clean_articles(wiki_text_directory, out_directory, dev_ids, test_ids):
    """
    Reads the articles from an extracted wikipedia dump and writes each paragraph to a separate file,
    with the following structure:

    out_directory
    ---| training
    -------| correct
    -----------| texts
    ---------------| 0000
    -------------------| <sequence_id>_<article_id>_<title>_<paragraph>.txt
    -------------------| ... (max 5000 files)
    ---------------| ...
    ---| development
    ---| test

    :param wiki_text_directory: directory of the wikipedia dump, containing folders with files containing articles
        as jsons
    :param out_directory: path where the paragraph files will be stored
    :param dev_ids: set of article IDs for the development set
    :param test_ids: set of article IDs for the test set
    """
    articles = get_article_jsons(wiki_text_directory)
    file_generator = BenchmarkFileNameGenerator(out_directory, "correct", dev_ids, test_ids)
    file_generator.prepare_directories()
    for article in articles:
        paragraphs = get_paragraphs(article)
        files = file_generator.get_sequence_files(article, len(paragraphs))
        for paragraph, file in zip(paragraphs, files):
            write_file(file, paragraph)


def write_dataset_split_files(wiki_text_directory, dev_ids, test_ids):
    """
    Reads the articles from an extracted wikipedia dump and writes three files, each containing the paragraphs
    of one partition.
    Also dumps three lists containing the sequences as Sequence-objects.
    The output file names are defined in src.settings.paths.

    :param wiki_text_directory: directory of the wikipedia dump, containing folders with files containing articles
        as jsons
    :param dev_ids: set of article IDs for the development set
    :param test_ids: set of article IDs for the test set
    :return:
    """
    articles = get_article_jsons(wiki_text_directory)

    training_file = open(paths.WIKI_TRAINING_FILE, 'wb')
    development_file = open(paths.WIKI_DEVELOPMENT_FILE, 'wb')
    test_file = open(paths.WIKI_TEST_FILE, 'wb')

    training_sequences = []
    development_sequences = []
    test_sequences = []

    sequence_id = 0
    for article in articles:
        article_id = article["id"]
        paragraphs = get_paragraphs(article)
        for paragraph in paragraphs:
            char_len = len(paragraph)
            bytes = (paragraph + '\n').encode(FILE_ENCODING)
            if article_id in dev_ids:
                byte_offset = development_file.tell()
                development_file.write(bytes)
                byte_len = development_file.tell() - byte_offset
                sequence = Sequence(sequence_id, DatasetSplit.TRAINING, byte_offset, byte_len, char_len)
                development_sequences.append(sequence)
            elif article_id in test_ids:
                byte_offset = test_file.tell()
                test_file.write(bytes)
                byte_len = test_file.tell() - byte_offset
                sequence = Sequence(sequence_id, DatasetSplit.DEVELOPMENT, byte_offset, byte_len, char_len)
                test_sequences.append(sequence)
            else:
                byte_offset = training_file.tell()
                training_file.write(bytes)
                byte_len = training_file.tell() - byte_offset
                sequence = Sequence(sequence_id, DatasetSplit.TEST, byte_offset, byte_len, char_len)
                training_sequences.append(sequence)
            sequence_id += 1

    training_file.close()
    development_file.close()
    test_file.close()
    del articles

    dump_object(training_sequences, paths.WIKI_TRAINING_SEQUENCES)
    dump_object(development_sequences, paths.WIKI_DEVELOPMENT_SEQUENCES)
    dump_object(test_sequences, paths.WIKI_TEST_SEQUENCES)


def count_paragraphs(wiki_text_directory):
    """
    Counts and prints the number of paragraphs of the dataset and number of paragraphs of the article with the most
    paragraphs. The latter is needed to define the sequence file naming convention, i.e. number of digits for the
    paragraph ID.

    :param wiki_text_directory: directory of the wikipedia dump, containing folders with files containing articles
        as jsons
    """
    n_paragraphs = 0
    n_articles = 0
    max_paragraphs_per_article = 0
    for article in get_article_jsons(wiki_text_directory):
        a_paragraphs = len(get_paragraphs(article))
        n_paragraphs += a_paragraphs
        n_articles += 1
        if a_paragraphs > max_paragraphs_per_article:
            max_paragraphs_per_article = a_paragraphs
    print("%i paragraphs in %i articles. Longest article has %i paragraphs." %
          (n_paragraphs, n_articles, max_paragraphs_per_article))


def detect_tables(wiki_text_directory):
    """
    Prints all paragraphs containing '|'.

    :param wiki_text_directory: directory of the wikipedia dump, containing folders with files containing articles
        as jsons
    """
    for article in get_article_jsons(wiki_text_directory):
        paragraphs = get_paragraphs(article)
        for para in paragraphs:
            if '|' in para:
                print(para)
