#!/usr/bin/env python3
import os
import random

from constants import DEFAULT_EVALUATOR_ALIGNEMENT, ROOT_PATH, BENCHMARKS_ENUM
from utils.multiviewer import MultiViewer
from utils.logger import logger
from utils.tolerant_comparer import is_correct_tolerant

MODELS_ENUM = [
    'beam_search_bwd_robust', 'beam_search_bwd', 'beam_search_labeling',
    'beam_search_robust', 'beam_search', 'labeling_noisy', 'labeling',
    'beam_search_labeling_robust', 'two_pass', 'two_pass_robust']


def read_triples(model, benchmark, evaluation_set='test', shuffle=False):
    assert model_a in MODELS_ENUM, 'model %s not in %s' % (model, MODELS_ENUM)
    assert model_b in MODELS_ENUM, 'model %s not in %s' % (model, MODELS_ENUM)
    assert benchmark in BENCHMARKS_ENUM, 'benchmark %s not in %s' % (benchmark, BENCHMARKS_ENUM)
    fixed = os.path.join(ROOT_PATH, 'results_sentences', benchmark,
                         evaluation_set, model + '.txt')
    correct = os.path.join(ROOT_PATH, 'benchmarks_sentences', benchmark,
                           evaluation_set, 'correct.txt')
    corrupt = os.path.join(ROOT_PATH, 'benchmarks_sentences', benchmark,
                           evaluation_set, 'corrupt.txt')
    with open(fixed, 'r') as fixed_file:
        with open(correct, 'r') as correct_file:
            with open(corrupt, 'r') as corrupt_file:
                triples = list(zip(correct_file, corrupt_file, fixed_file))
    if shuffle:
        random.seed(41)
        random.shuffle(triples)
    return triples


def get_original_sentences():
    with open(ROOT_PATH + "sentences/test.txt", "r") as f:
        lines = f.readlines()
    lines = [line[:-1] for line in lines]
    return lines


def compare_two(original, data_a, data_b, count):
    correct_a, corrupt_a, fixed_a, benchmark_a, model_a = data_a
    correct_b, corrupt_b, fixed_b, benchmark_b, model_b = data_b

    metrics_a, out_a = comparator.evaluate(correct_a, corrupt_a, fixed_a)
    metrics_b, out_b = comparator.evaluate(correct_b, corrupt_b, fixed_b)
    #if metrics_a[-1] == metrics_b[-1]: return 0
    logger.output(original)
    logger.log_info(model_b, benchmark_b, highlight=3)
    logger.output(out_b)

    logger.output(original)
    logger.log_info(model_a, benchmark_a, highlight=2)
    logger.output(out_a)

    logger.log_info(model_a, 'is better', highlight=5)
    logger.log_info(count, 'errors so far')
    logger.log_seperator()
    logger.log_seperator()
    return 1


if __name__ == '__main__':
    model_a = 'beam_search_labeling_robust'#'two_pass_robust'
    model_b = 'beam_search_labeling'
    benchmark_a = '0_inf' #'0.1_0.1'
    benchmark_b = '0_inf'

    original_sentences = get_original_sentences()
    comparator = MultiViewer()

    a, b = 0, 0
    for original, (correct_a, corrupt_a, fixed_a), (correct_b, corrupt_b, fixed_b) in zip(
            original_sentences, read_triples(model_a, benchmark_a), read_triples(model_b, benchmark_b)):
        if (not is_correct_tolerant(original, correct_a, corrupt_a, fixed_a) and
                is_correct_tolerant(original, correct_b, corrupt_b, fixed_b)):
            b += 1
        if (is_correct_tolerant(original, correct_a, corrupt_a, fixed_a) and not
                is_correct_tolerant(original, correct_b, corrupt_b, fixed_b)):
            a += 1
    logger.log_info("%d where %s is better, %d where %s is better" % (
        a, model_a, b, model_b), highlight=1)

    count = 0
    for original, (correct_a, corrupt_a, fixed_a), (correct_b, corrupt_b, fixed_b) in zip(
            original_sentences, read_triples(model_a, benchmark_a), read_triples(model_b, benchmark_b)):
        # B is better, A is bad
        if (not is_correct_tolerant(original, correct_a, corrupt_a, fixed_a) and
                is_correct_tolerant(original, correct_b, corrupt_b, fixed_b)):
            data_a = (correct_a, corrupt_a, fixed_a, benchmark_a, model_a)
            data_b = (correct_b, corrupt_b, fixed_b, benchmark_b, model_b)
            count += compare_two(original, data_b, data_a, count)
    count = 0
    for original, (correct_a, corrupt_a, fixed_a), (correct_b, corrupt_b, fixed_b) in zip(
            original_sentences, read_triples(model_a, benchmark_a), read_triples(model_b, benchmark_b)):
        # A is better, B is bad
        if (is_correct_tolerant(original, correct_a, corrupt_a, fixed_a) and not
                is_correct_tolerant(original, correct_b, corrupt_b, fixed_b)):
            data_a = (correct_a, corrupt_a, fixed_a, benchmark_a, model_a)
            data_b = (correct_b, corrupt_b, fixed_b, benchmark_b, model_b)
            count += compare_two(original, data_a, data_b, count)
