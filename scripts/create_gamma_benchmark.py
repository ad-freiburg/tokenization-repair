from typing import List
import sys
from scipy.stats import gamma
import random

import project
from src.helper.files import read_lines
from src.noise.acl_noise_inducer import ACLNoiseInducer
from src.helper.stochastic import flip_coin


def read_data(path: str) -> List[float]:
    lines = read_lines(path)
    data = [float(line) for line in lines]
    return data


def fit_gamma(data):
    return gamma.fit(data)


def sample_gamma(params):
    a, loc, scale = params
    return gamma.rvs(a, loc=loc, scale=scale)


def corrupt_tokenization(sequence: str, p_space: float):
    corrupt = ""
    for pos in range(len(sequence)):
        if pos + 1 < len(sequence) and sequence[pos + 1] != " " and flip_coin(random, p_space):
            if sequence[pos] != " ":
                corrupt += sequence[pos] + " "
        else:
            corrupt += sequence[pos]
    return corrupt


def create_sequence_spans(sequence: str) -> List[str]:
    tokens = sequence.split(" ")
    spans = []
    pos = 0
    while pos < len(tokens):
        n_tokens = random.randint(5, 10)
        spans.append(" ".join(tokens[pos:(pos + n_tokens)]))
        pos += n_tokens
    return spans


def recombine_spans(spans: List[str]) -> str:
    return " ".join(spans)


if __name__ == "__main__":
    in_file = sys.argv[1]
    ocr_error_rates_file = sys.argv[2]
    tokenization_error_rates_file = sys.argv[3]
    out_dir = sys.argv[4]
    seed = 20210424

    random.seed(seed)

    ocr_error_rates = read_data(ocr_error_rates_file)
    tokenization_error_rates = read_data(tokenization_error_rates_file)
    gamma_params_ocr = fit_gamma(ocr_error_rates)
    print("ocr", gamma_params_ocr)
    gamma_params_tokenization = fit_gamma(tokenization_error_rates)
    print("tokenization", gamma_params_tokenization)
    ocr_noise_inducer = ACLNoiseInducer(p=0, insertion_prob=0.2079, seed=seed)

    with open(out_dir + "/correct.txt", "w") as correct_file, open(out_dir + "/corrupt.txt", "w") as corrupt_file:
        for sequence in read_lines(in_file):
            print(sequence)
            spans = create_sequence_spans(sequence)
            misspelled_spans = []
            mistokenized_spans = []
            for span in spans:
                p_ocr = sample_gamma(gamma_params_ocr)
                ocr_noise_inducer.p = p_ocr
                p_space = sample_gamma(gamma_params_tokenization)
                misspelled = ocr_noise_inducer.induce_noise(span)
                mistokenized = corrupt_tokenization(misspelled, p_space)
                misspelled_spans.append(misspelled)
                mistokenized_spans.append(mistokenized)
            misspelled = recombine_spans(misspelled_spans)
            mistokenized = recombine_spans(mistokenized_spans)
            print(misspelled)
            print(mistokenized)
            correct_file.write(misspelled + "\n")
            corrupt_file.write(mistokenized + "\n")
