#!/usr/bin/env python3
import os
import random

from utils.multiviewer import MultiViewer
from utils.logger import logger


BENCHMARKS_ENUM = ['0_0.1', '0_1', '0.1_0.1', '0.1_1', '0.1_inf', '0_inf']
ROOT_PATH = '/nfs/students/matthias-hertel/tokenization-repair-paper/'
MODELS_ENUM = [
    'beam_search_bwd_robust', 'beam_search_bwd', 'beam_search_labeling',
    'beam_search_robust', 'beam_search', 'labeling_noisy', 'labeling', 
    'two_pass']


def read_triples(model_a, model_b, benchmark, evaluation_set='test', shuffle=False):
    assert model_a in MODELS_ENUM, 'model %s not in %s' % (model, MODELS_ENUM)
    assert model_b in MODELS_ENUM, 'model %s not in %s' % (model, MODELS_ENUM)
    assert benchmark in BENCHMARKS_ENUM, 'benchmark %s not in %s' % (benchmark, BENCHMARKS_ENUM)
    fixed_a = os.path.join(ROOT_PATH, 'results_sentences', benchmark,
                           evaluation_set, model_a + '.txt')
    fixed_b = os.path.join(ROOT_PATH, 'results_sentences', benchmark,
                           evaluation_set, model_b + '.txt')
    correct = os.path.join(ROOT_PATH, 'benchmarks_sentences', benchmark,
                           evaluation_set, 'correct.txt')
    corrupt = os.path.join(ROOT_PATH, 'benchmarks_sentences', benchmark,
                           evaluation_set, 'corrupt.txt')
    with open(fixed_a, 'r') as fixed_file_a:
        with open(fixed_b, 'r') as fixed_file_b:
            with open(correct, 'r') as correct_file:
                with open(corrupt, 'r') as corrupt_file:
                    triples = list(zip(correct_file, corrupt_file, fixed_file_a, fixed_file_b))
    if shuffle:
        random.seed(41)
        random.shuffle(triples)
    return triples


if __name__ == '__main__':
    model_a = 'two_pass'
    model_b = 'labeling'
    benchmark = '0_0.1'
    take_first_n = 100
    comparator = MultiViewer()

    for correct, corrupt, fixed_a, fixed_b in read_triples(model_a, model_b, benchmark):
        if not ((correct == fixed_a) ^ (correct == fixed_b)):
            continue
        if take_first_n < 1:
            break
        _, out_a = comparator.evaluate(correct, corrupt, fixed_a)
        _, out_b = comparator.evaluate(correct, corrupt, fixed_b)
        logger.log_info(model_a, benchmark, highlight=2)
        logger.output(out_a)
        logger.log_info(model_b, benchmark, highlight=3)
        logger.output(out_b)
        logger.log_seperator()
        take_first_n -= 1
