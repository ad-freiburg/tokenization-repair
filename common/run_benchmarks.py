#!/usr/bin/env python3
from optparse import OptionParser

from configs import get_fixer_config
from constants import FIXERS_ENUM
from handlers.benchmark import Benchmark
from utils.logger import logger


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
        config = get_fixer_config(fixer=FIXERS_ENUM.bicontext_fixer, use_look_forward=True)
    if dpfixer:
        config = get_fixer_config(fixer=FIXERS_ENUM.dp_fixer)

    benchmark = Benchmark(config)
    benchmark.run()
