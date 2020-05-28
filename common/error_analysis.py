import os
import random

from utils.multiviewer import MultiViewer


BENCHMARKS_ENUM = ['0_0.1', '0_1', '0.1_0.1', '0.1_1', '0.1_inf', '0_inf']
ROOT_PATH = '/nfs/students/matthias-hertel/tokenization-repair-paper/'
MODELS_ENUM = [
    'beam_search_bwd_robust', 'beam_search_bwd', 'beam_search_labeling',
    'beam_search_robust', 'beam_search', 'labeling_noisy', 'labeling', 
    'two_pass']


def read_triples(model, benchmark, evaluation_set='test', shuffle=False):
    assert model in MODELS_ENUM, 'model %s not in %s' % (model, MODELS_ENUM)
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


if __name__ == '__main__':
    model = 'beam_search_bwd_robust'
    benchmark = '0_0.1'
    take_first_n = 100
    comparator = MultiViewer()

    for correct, corrupt, fixed in read_triples(model, benchmark):
        if take_first_n < 1:
            break
        metrics, out = comparator.evaluate(correct, corrupt, fixed)
        accuracy = metrics[-1]
        if accuracy < 1:
            print(out)
            take_first_n -= 1
