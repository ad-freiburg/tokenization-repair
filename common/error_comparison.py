#!/usr/bin/env python3
import os
import random

from utils.multiviewer import MultiViewer
from utils.logger import logger
from utils.tolerant_comparer import print_comparison, is_correct_tolerant

BENCHMARKS_ENUM = ['0_0.1', '0_1', '0.1_0.1', '0.1_1', '0.1_inf', '0_inf']
ROOT_PATH = '/nfs/students/matthias-hertel/tokenization-repair-paper/'
MODELS_ENUM = [
    'beam_search_bwd_robust', 'beam_search_bwd', 'beam_search_labeling',
    'beam_search_robust', 'beam_search', 'labeling_noisy', 'labeling', 
    'two_pass', 'two_pass_robust']


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


def get_original_sentences():
    with open(ROOT_PATH + "sentences/test.txt", "r") as f:
        lines = f.readlines()
    lines = [line[:-1] for line in lines]
    return lines

if __name__ == '__main__':
    model_a = 'two_pass'
    model_b = 'two_pass_robust'
    benchmark = '0.1_1'  #'0_0.1'
    take_first_n = 100
    count = 0
    a, b = 0, 0
    original_sentences = get_original_sentences()
    i = 0
    for correct, corrupt, fixed_a, fixed_b in read_triples(model_a, model_b, benchmark):
        original = original_sentences[i]
        cond_b = is_correct_tolerant(original, correct, corrupt, fixed_b)
        cond_a = is_correct_tolerant(original, correct, corrupt, fixed_a)
        if cond_a and cond_b:
            continue
        if not cond_a and not cond_b:
            continue
        if cond_a and not cond_b:
            a += 1
        if cond_b and not cond_a:
            b += 1
    logger.log_info("superior times", a, b)

    comparator = MultiViewer()
    i = 0
    for correct, corrupt, fixed_a, fixed_b in read_triples(model_a, model_b, benchmark):
        original = original_sentences[i]
        i += 1
        if not (is_correct_tolerant(original, correct, corrupt, fixed_b) and
                not is_correct_tolerant(original, correct, corrupt, fixed_a)):
            continue
        if take_first_n < 1:
            break
        metrics_a, out_a = comparator.evaluate(correct, corrupt, fixed_a)
        metrics_b, out_b = comparator.evaluate(correct, corrupt, fixed_b)
        acc_a = metrics_a[-1]
        acc_b = metrics_b[-1]
        if acc_a < acc_b:
            logger.output(' ' * 120, original)
            logger.log_info(model_a, benchmark, highlight=2)
            #print_comparison(original, correct, corrupt, fixed_a)
            logger.output(out_a)
            logger.log_info(model_b, benchmark, highlight=3)
            #print_comparison(original, correct, corrupt, fixed_b)
            logger.output(out_b)
            logger.log_info(model_b, 'is better', highlight=5)
            count += 1
            logger.log_info(count, 'errors so far')
            logger.log_seperator()
            #take_first_n -= 1
    i = 0
    take_first_n = 100
    for correct, corrupt, fixed_a, fixed_b in read_triples(model_a, model_b, benchmark):
        original = original_sentences[i]
        i += 1
        #if not ((correct == fixed_a) ^ (correct == fixed_b)):
        #    continue
        if not (is_correct_tolerant(original, correct, corrupt, fixed_a) and
                not is_correct_tolerant(original, correct, corrupt, fixed_b)):
            continue
        if take_first_n < 1:
            break
        metrics_a, out_a = comparator.evaluate(correct, corrupt, fixed_a)
        metrics_b, out_b = comparator.evaluate(correct, corrupt, fixed_b)
        acc_a = metrics_a[-1]
        acc_b = metrics_b[-1]
        if acc_b < acc_a:
            #logger.output(original)
            logger.output(' ' * 120, original)
            logger.log_info(model_a, benchmark, highlight=2)
            #print_comparison(original, correct, corrupt, fixed_a)
            logger.output(out_a)
            logger.log_info(model_b, benchmark, highlight=3)
            #print_comparison(original, correct, corrupt, fixed_b)
            logger.output(out_b)
            logger.log_info(model_a, 'is better', highlight=5)
            count += 1
            logger.log_info(count, 'errors so far')
            logger.log_seperator()
            #take_first_n -= 1
