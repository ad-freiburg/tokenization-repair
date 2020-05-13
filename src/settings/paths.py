from typing import Optional

from src.helper.files import path_exists, make_directory
from src.benchmark.subset import Subset

# BASE DIRECTORY
DUMP_DIRS = [
    "/mnt/4E83539B6CE67342/tokenization-repair-dumps/",  # Lappy
    "/project/master/hertelm/tokenization-repair-dumps/",  # tfpools
    "/local/hdd/exports/data/matthias-hertel/tokenization-repair-dumps/",  # titan
    "/local/data/hertelm/tokenization-repair-dumps/",  # sirba
    "/nfs/students/matthias-hertel/tokenization-repair-dumps/",  # ad
    "/data/1/matthias-hertel/tokenization-repair-dumps/",  # cluster
    "/home/hertel/tokenization-repair-dumps/",  # wunderfitz
    "/external/"  # docker
]
DUMP_DIR = None
for dir in DUMP_DIRS:
    if path_exists(dir):
        DUMP_DIR = dir
        print("Located dump folder: %s" % DUMP_DIR)
        break
if DUMP_DIR is None:
    raise Exception("Unable to locate dump folder.")

# MODEL DIRECTORY FOR SERVER
MODEL_FOLDER = DUMP_DIR + "models_server/"

# ESTIMATOR DIRECTORY
ESTIMATORS_DIR = DUMP_DIR + "estimators/"

# DATA DIRECTORY
DATA_DIR = DUMP_DIR + "data/"

# WIKIPEDIA DIRECTORY
WIKI_DIRS = ["/mnt/4E83539B6CE67342/tokenization-repair-dumps/data/wikipedia/",
             "/project/master/hertelm/wikipedia/",
             "/local/hdd/exports/data/matthias-hertel/wikipedia/",
             "/local/data/hertelm/wikipedia/",
             "/data/1/matthias-hertel/wikipedia/",
             "/home/hertel/wikipedia/"]
WIKI_DIR = None
for dir in WIKI_DIRS:
    if path_exists(dir):
        WIKI_DIR = dir
        break
if WIKI_DIR is None:
    WIKI_DIR = "__UNKNOWN_WIKI_DIRECTORY__"
    print("WARNING: Unable to locate wikipedia folder.")

# wikipedia article IDs
WIKI_TRAINING_ARTICLE_IDS = WIKI_DIR + "training_article_ids.pkl"
WIKI_DEVELOPMENT_ARTICLE_IDS = WIKI_DIR + "development_article_ids.pkl"
WIKI_TEST_ARTICLE_IDS = WIKI_DIR + "test_article_ids.pkl"

# paragraphs
WIKI_PARAGRAPHS_DIR = WIKI_DIR + "single-file/"
WIKI_TRAINING_PARAGRAPHS = WIKI_PARAGRAPHS_DIR + "training_shuffled.txt"

# sentence files
WIKI_SENTENCES_DIR = WIKI_DIR + "sentences/"
WIKI_TRAINING_SENTENCES = WIKI_SENTENCES_DIR + "training.txt"
WIKI_TRAINING_SENTENCES_SHUFFLED = WIKI_SENTENCES_DIR + "training_shuffled.txt"
WIKI_TUNING_SENTENCES = WIKI_SENTENCES_DIR + "tuning.txt"
WIKI_DEVELOPMENT_SENTENCES = WIKI_SENTENCES_DIR + "development.txt"
WIKI_TEST_SENTENCES = WIKI_SENTENCES_DIR + "test.txt"

# punkt tokenizer
WIKI_PUNKT_TOKENIZER = WIKI_DIR + "punkt_tokenizer.pkl"
EXTENDED_PUNKT_ABBREVIATIONS = WIKI_DIR + "extended_punkt_abbreviations.pkl"

# BENCHMARKS DIRECTORY
BENCHMARKS_DIR = DUMP_DIR + "benchmarks_sentences/"

# RESULTS DIRECTORY
RESULTS_DIR = DUMP_DIR + "results_sentences/"
RESULTS_DICT = RESULTS_DIR + "results.pkl"

# PLOTS DIRECTORY
PLOT_DIR = DUMP_DIR + "plots/"

# DICTIONARY DIRECTORY
DICT_FOLDER = DUMP_DIR + "dictionaries/"
# character frequencies
CHARACTER_FREQUENCY_DICT = DICT_FOLDER + "character_frequencies.pkl"
# old dictionaries for reproducibility
WIKI_ENCODER_DICT = DICT_FOLDER + "char2ix.pkl"
WIKI_DECODER_DICT = DICT_FOLDER + "ix2char.pkl"
# decision thresholds
DECISION_THRESHOLD_FILE = DICT_FOLDER + "new_decision_thresholds.pkl"
SINGLE_RUN_DECISION_THRESHOLD_FILE = DICT_FOLDER + "single_run_decision_thresholds.pkl"
# beam search penalties
BEAM_SEARCH_PENALTY_FILE = DICT_FOLDER + "beam_search_penalties.pkl"
# token frequencies
TOKEN_FREQUENCY_DICT = DICT_FOLDER + "token_frequencies.pkl"
# unigrams
UNIGRAM_DELIM_FREQUENCY_DICT = DICT_FOLDER + "unigram_delim_frequencies.pkl"
UNIGRAM_NO_DELIM_FREQUENCY_DICT = DICT_FOLDER + "unigram_no_delim_frequencies.pkl"
MOST_FREQUENT_UNIGRAMS_DICT = DICT_FOLDER + "unigrams_most_frequent_%i.pkl"
# bigrams
BIGRAM_HOLDER = DICT_FOLDER + "bigram_holder.pkl"
# stump dict
STUMP_DICT = DICT_FOLDER + "token_stumps.pkl"

# INTERMEDIATE RESULTS DIRECTORY
INTERMEDIATE_DIR = DUMP_DIR + "intermediate/"
THRESHOLD_FITTER_DIR = INTERMEDIATE_DIR + "threshold_fitter/"

# OPENAI
OPENAI_MODELS_FOLDER = None

for dir in [RESULTS_DIR, PLOT_DIR, INTERMEDIATE_DIR, THRESHOLD_FITTER_DIR, BENCHMARKS_DIR]:
    if not path_exists(dir):
        make_directory(dir)
        print("Made directory: %s" % dir)


def benchmark_sub_directory(name: str,
                            subset: Subset,
                            subfolder: Optional[str]) -> str:
    subdir = name + "/" + subset.folder_name() + "/"
    if subfolder is not None:
        subdir += subfolder + "/"
    return subdir
