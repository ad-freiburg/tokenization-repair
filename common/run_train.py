#!/usr/bin/env python3
import os
import sys
import warnings

from optparse import OptionParser

from constants import DEFAULT_DATA_LOAD_DIR, BACKWARD
from configs import E2E_MODES_ENUM, FIXERS_ENUM, MODELS_ENUM
from handlers.reader import Reader
from utils.logger import logger

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import nltk
# emma = nltk.corpus.gutenberg.raw('burgess-busterbrown.txt')
# texts = ['Hello world this is boring', 'Attack on Titan is cool', emma[:60000]]
# evals = [emma[60000:]]
subset = 10 # None


def debug_reader(**kwargs):
    from configs import get_language_model_config
    config = get_language_model_config(inference=True, **kwargs)
    # model.sample_analysis()
    reader = Reader(config)
    steps, gen = reader.read_development_text_onefile(num_lines_per_epoch=10)
    for i, ln in enumerate(gen):
        if i > 100: break


def debug_triples_reader(**kwargs):
    from configs import get_language_model_config
    config = get_language_model_config(inference=True, **kwargs)
    # model.sample_analysis()
    reader = Reader(config)
    gen = reader.read_development_triples_onefile()
    for i, x in enumerate(gen):
        if i > 1000: break
        # logger.log_debug(x, highlight=3)


def train_language_model(**kwargs):
    from configs import get_language_model_config
    config = get_language_model_config(**kwargs)
    # model.sample_analysis()
    reader = Reader(config)
    """
    correct_texts = reader.read_train_texts()
    val_correct_texts = reader.read_test_texts()
    if subset is not None:
        correct_texts = correct_texts[:subset]
        val_correct_texts = val_correct_texts[:subset]
    logger.log_debug(correct_texts[:10])
    logger.log_debug(val_correct_texts[:10])
    logger.log_debug(os.listdir(DEFAULT_DATA_LOAD_DIR))
    """
    #steps, generator = reader.read_test_text_onefile()
    steps, generator = reader.read_train_text_onefile()
    valid_steps, valid_generator = reader.read_development_text_onefile(
        backward=(config.direction == BACKWARD))
    from models.mostafa.rnn_language_model import RNNLanguageModel
    model = RNNLanguageModel(config)
    model.train(generator, steps, valid_generator, valid_steps)


def train_e2e_language_model(**kwargs):
    from models.mostafa.rnn_e2e_bidirectional_model import E2eBidirectionLanguageModel
    from configs import get_decision_model_config
    config = get_decision_model_config(**kwargs)
    model = E2eBidirectionLanguageModel(config)
    reader = Reader(config)
    """
    train_data = reader.read_train_decision_triples()
    val_data = reader.read_test_decision_triples()
    if subset is not None:
        train_data = train_data[:subset]
        val_data = val_data[:subset // 10]
    """
    train_data = reader.read_train_triples_onefile()
    val_data = reader.read_development_triples_onefile()
    model.train(train_data, val_data)


def train_tuner(**kwargs):
    from models.mostafa.bicontext_tuner import BicontextTuner
    from configs import get_fixer_config
    config = get_fixer_config(fixer=FIXERS_ENUM.bicontext_fixer, **kwargs)

    tuner = BicontextTuner(config)
    reader = Reader(config)
    #train_data = reader.read_train_decision_triples(sort_by_size=True)
    #if subset is not None:
    #    train_data = train_data[:subset]
    # train_data = reader.read_train_triples_onefile()
    train_data = reader.read_train_pairs_onefile()
    tuner.train(train_data)


def train_all():
    combinations = [
        #{'lf': 5, 'decision_units': 8, 'probs_units': 32, 'decision_layers': 1, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 32, 'decision_layers': 2, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 32, 'decision_layers': 3, 'use_look_forward': True, 'epochs': 120},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 32, 'decision_layers': 1, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 32, 'decision_layers': 2, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 32, 'decision_layers': 3, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 32, 'decision_layers': 1, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 32, 'decision_layers': 2, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 32, 'decision_layers': 3, 'use_look_forward': True, 'learning_rate': 0.02},
        # {'lf': 5, 'decision_units': 8, 'probs_units': 64, 'decision_layers': 1, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 64, 'decision_layers': 2, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 64, 'decision_layers': 3, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 64, 'decision_layers': 1, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 64, 'decision_layers': 2, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 64, 'decision_layers': 3, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 64, 'decision_layers': 1, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 64, 'decision_layers': 2, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 64, 'decision_layers': 3, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 16, 'decision_layers': 1, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 16, 'decision_layers': 2, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 16, 'decision_layers': 3, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 16, 'decision_layers': 1, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 16, 'decision_layers': 2, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 16, 'decision_layers': 3, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 16, 'decision_layers': 1, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 16, 'decision_layers': 2, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 32, 'probs_units': 16, 'decision_layers': 3, 'use_look_forward': True, 'learning_rate': 0.01},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 16, 'decision_layers': 3, 'use_look_forward': True},
        #{'lf': 5, 'decision_units': 16, 'probs_units': 16, 'decision_layers': 1, 'use_look_forward': True},
        {'lf': 5, 'decision_units': 16, 'probs_units': 16, 'decision_layers': 1, 'use_look_forward': True, 'learning_rate': 0.2,
         'topk': 1},
        #{'lf': 5, 'decision_units': 8, 'probs_units': 16, 'decision_layers': 3, 'use_look_forward': False},
    ]
    for kwargs in combinations:
        train_e2e_language_model(**kwargs)


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
    parser.add_option("-e", "--e2e-decision", dest="e2e_decision",
                      action="store_true",
                      help="train end-to-end decision model",
                      metavar="E2ERNN")
    parser.add_option("-l", "--e2e-lang", dest="e2e_language",
                      action="store_true",
                      help="train end-to-end language model",
                      metavar="E2ERNN")
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
    e2e_language = options.e2e_language is not None
    e2e_decision = options.e2e_decision is not None

    logger.set_verbose(verbose)
    logger.set_debug(debug)
    logger.set_full_debug(full_debug)

    # debug_triples_reader(); sys.exit(0)
    # debug_reader(); sys.exit(0)
    if forward_rnn:
        train_language_model(model=MODELS_ENUM.forward_language_model, epochs=10,
                             rnn_layers=3, rnn_units=256)
        train_language_model(model=MODELS_ENUM.forward_language_model, epochs=10,
                             rnn_layers=2, rnn_units=512)
    if backward_rnn:
        #train_language_model(model=MODELS_ENUM.backward_language_model, epochs=5)
        train_language_model(model=MODELS_ENUM.backward_language_model, epochs=10,
                             rnn_layers=2, rnn_units=512, batch_size=2048)
        train_language_model(model=MODELS_ENUM.backward_language_model, epochs=10,
                             rnn_layers=3, rnn_units=256, batch_size=4096)
    if e2e_language:
        train_e2e_language_model(model=MODELS_ENUM.e2e_model,
                                 mode=E2E_MODES_ENUM.language, epochs=5)
    if e2e_decision:
        train_e2e_language_model(model=MODELS_ENUM.e2e_model,
                                 # mode=E2E_MODES_ENUM.decision)
                                 mode=E2E_MODES_ENUM.full_e2e, epochs=10)
    if tuner:
        train_tuner()  # epochs=10000)
    # train_all()
