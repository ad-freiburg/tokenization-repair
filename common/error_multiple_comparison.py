#!/usr/bin/env python3
import os
import random

from constants import DEFAULT_EVALUATOR_ALIGNEMENT
from utils.multiviewer import MultiViewer
from utils.logger import logger
from utils.tolerant_comparer import print_comparison, is_correct_tolerant

BENCHMARKS_ENUM = ['0_0.1', '0_1', '0.1_0.1', '0.1_1', '0.1_inf', '0_inf']
ROOT_PATH = '/nfs/students/matthias-hertel/tokenization-repair-paper/'
MODELS_ENUM = [
    'beam_search_bwd_robust', 'beam_search_bwd', 'beam_search_labeling',
    'beam_search_robust', 'beam_search', 'labeling_noisy', 'labeling', 
    'two_pass', 'two_pass_robust']


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


def compare_two(original, data_a, data_b, count, better_model):
    correct_a, corrupt_a, fixed_a, benchmark_a, model_a = data_a
    correct_b, corrupt_b, fixed_b, benchmark_b, model_b = data_b

    metrics_a, out_a = comparator.evaluate(correct_a, corrupt_a, fixed_a)
    metrics_b, out_b = comparator.evaluate(correct_b, corrupt_b, fixed_b)
    logger.output(original)
    logger.log_info(model_a, benchmark_a, highlight=2)
    logger.output(out_a)
    logger.log_info(model_b, benchmark_b, highlight=3)
    logger.output(out_b)
    logger.log_info(better_model, 'is better', highlight=5)
    logger.log_info(count, 'errors so far')
    logger.log_seperator()


if __name__ == '__main__':
    model_a = 'two_pass'
    model_b = 'two_pass_robust'
    benchmark_a = '0.1_1'  #'0_0.1'
    benchmark_b = '0_0.1'
    count = 0
    a, b = 0, 0
    original_sentences = get_original_sentences()

    comparator = MultiViewer()
    i = 0
    count = 0
    for (correct_a, corrupt_a, fixed_a), (correct_b, corrupt_b, fixed_b) in zip(read_triples(model_a, benchmark_a), read_triples(model_b, benchmark_b)):
        original = original_sentences[i]
        i += 1
        # A is better
        if not (is_correct_tolerant(original, correct_a, corrupt_a, fixed_a) and
                not is_correct_tolerant(original, correct_b, corrupt_b, fixed_b)):
            continue
        count += 1
        data_a = (correct_a, corrupt_a, fixed_a, benchmark_a, model_a)
        data_b = (correct_b, corrupt_b, fixed_b, benchmark_b, model_b)
        compare_two(original, data_a, data_b, count, model_a)
    i = 0
    count = 0
    for (correct_a, corrupt_a, fixed_a), (correct_b, corrupt_b, fixed_b) in zip(read_triples(model_a, benchmark_a), read_triples(model_b, benchmark_b)):
        original = original_sentences[i]
        i += 1
        if not (not is_correct_tolerant(original, correct_a, corrupt_a, fixed_a) and
                is_correct_tolerant(original, correct_b, corrupt_b, fixed_b)):
            continue
        count += 1
        data_a = (correct_a, corrupt_a, fixed_a, benchmark_a, model_a)
        data_b = (correct_b, corrupt_b, fixed_b, benchmark_b, model_b)
        compare_two(original, data_a, data_b, count, model_b)
