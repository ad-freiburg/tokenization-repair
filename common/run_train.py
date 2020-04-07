#!/usr/bin/env python3
import os
import sys
import warnings

from optparse import OptionParser

from configs import MODELS_ENUM, get_language_model_config, get_bicontext_fixer_config
from handlers.reader import Reader
from utils.logger import logger
from models.rnn_bicontext_fixer import Tuner
from models.rnn_language_model import RNNLanguageModel

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import nltk
# emma = nltk.corpus.gutenberg.raw('burgess-busterbrown.txt')
# texts = ['Hello world this is boring', 'Attack on Titan is cool', emma[:60000]]


def train_language_model(**kwargs):
    config = get_language_model_config(**kwargs)
    reader = Reader(config)
    model = RNNLanguageModel(config)
    model.train(reader.read_train_lines())


def train_tuner(**kwargs):
    config = get_bicontext_fixer_config(**kwargs)
    reader = Reader(config)
    tuner = Tuner(config)
    tuner.train_data(reader.read_valid_pairs(), total=30)


if __name__ == '__main__':
    parser = OptionParser(("run_train.py [options]\n"
                           "to train one of the following models"))
    parser.add_option("-v", "--verbose", dest="verbose", action='store_true',
                      help="print update statements during computations",
                      metavar="VERBOSE")
    parser.add_option("-x", "--debug", dest="debug", action='store_true',
                      help="print update debug statements during computations",
                      metavar="DEBUG")
    parser.add_option("-t", "--tuner", dest="tuner",
                      action="store_true",
                      help="train bicontext tuner",
                      metavar="TUNER")
    parser.add_option("-b", "--back-rnn", dest="backward_rnn",
                      action="store_true",
                      help="train backward language model",
                      metavar="BRNN")
    parser.add_option("-f", "--forw-rnn", dest="forward_rnn",
                      action="store_true",
                      help="train forward language model",
                      metavar="BRNN")
    parser.add_option("-z", "--full-debug", dest="full_debug",
                      action='store_true',
                      help="print full debug statements during computations",
                      metavar="FULLDEBUG")

    options, args = parser.parse_args()
    verbose = options.verbose is not None
    debug = options.debug is not None
    full_debug = options.full_debug is not None

    tuner = options.tuner is not None
    forward_rnn = options.forward_rnn is not None
    backward_rnn = options.backward_rnn is not None
    tuner = options.tuner is not None

    logger.set_verbose(verbose)
    logger.set_debug(debug)
    logger.set_full_debug(full_debug)

    if forward_rnn:
        train_language_model(model=MODELS_ENUM.forward_language_model)
    if backward_rnn:
        train_language_model(model=MODELS_ENUM.backward_language_model)
    if tuner:
        train_tuner()
