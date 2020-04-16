#!/usr/bin/env python3
import os
import warnings
import _pickle as pickle

from optparse import OptionParser

from configs import get_fixer_config
from constants import FIXERS_ENUM
from handlers.benchmark import Benchmark
from handlers.reader import Reader
from models.rnn_bicontext_fixer import Tuner
from utils.logger import logger

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def export_dict(config):
    from models.trie_dictionary import TrieDictionary
    if not os.path.isfile(config.dictionary_path):
        dictionary = TrieDictionary(config)
        with open(config.dictionary_path, 'wb') as fl:
            pickle.dump(dictionary, fl)
            logger.log_info('exported dictionary into', config.dictionary_path)


def train_tuner(config):
    reader = Reader(config)
    tuner = Tuner(config)
    tuner.train_data(reader.read_valid_pairs(), total=550)


if __name__ == '__main__':
    parser = OptionParser(("run_benchmarks.py [options]\n"
                           "Run the benchmarks on one dataset using one model"
                           ))
    parser.add_option("-v", "--verbose", dest="verbose", action='store_true',
                      help="print update statements during computations",
                      metavar="VERBOSE")
    parser.add_option("-x", "--debug", dest="debug", action='store_true',
                      help="print update debug statements during computations",
                      metavar="DEBUG")
    parser.add_option("-z", "--full-debug", dest="full_debug",
                      action='store_true',
                      help="print full debug statements during computations",
                      metavar="FULLDEBUG")
    parser.add_option("-d", "--dp", dest="dpfixer", action="store_true",
                      help='Fix using baseline dp model',
                      metavar="E2ERNN")
    parser.add_option("-c", "--bicontext", dest="bicontext", action="store_true",
                      help='Fix using bicontext model',
                      metavar="BICONTEXT")

    options, args = parser.parse_args()
    verbose = options.verbose is not None
    debug = options.debug is not None
    full_debug = options.full_debug is not None
    dpfixer = options.dpfixer is not None
    bicontext = options.bicontext is not None

    logger.set_verbose(verbose)
    logger.set_debug(debug)
    logger.set_full_debug(full_debug)

    if bicontext:
        config = get_fixer_config(fixer=FIXERS_ENUM.bicontext_fixer)
        train_tuner(config)
    if dpfixer:
        config = get_fixer_config(fixer=FIXERS_ENUM.dp_fixer)
        export_dict(config)

    benchmark = Benchmark(config)
    benchmark.run()
