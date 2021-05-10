from typing import List
import sys
import scipy.stats
import random
from hyphen import Hyphenator

import project
from src.helper.files import read_lines
from src.noise.acl_noise_inducer import ACLNoiseInducer
from src.helper.stochastic import flip_coin


def read_data(path: str) -> List[float]:
    lines = read_lines(path)
    data = [float(line) for line in lines]
    return data


def fit_distribution(data, distribution, fscale=None):
    if fscale is None:
        return distribution.fit(data)
    else:
        return distribution.fit(data, fscale=fscale)


def sample_distribution(params, distribution):
    a, loc, scale = params
    return distribution.rvs(a, loc=loc, scale=scale)


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


class HyphenationIntroducer:
    def __init__(self, p_hyphen: float):
        self.p_hyphen = p_hyphen
        self.hyphenator = Hyphenator()

    def get_candidates(self, token: str) -> List[str]:
        try:
            return self.hyphenator.pairs(token)
        except:
            return []

    def introduce_hyphens(self, text: str) -> str:
        tokens = text.split(" ")
        for i in range(len(tokens)):
            candidates = self.get_candidates(tokens[i])
            if len(candidates) > 0 and flip_coin(random, self.p_hyphen):
                candidate = random.choice(candidates)
                tokens[i] = "-".join(candidate)
        return " ".join(tokens)


if __name__ == "__main__":
    in_file = sys.argv[1]
    ocr_error_rates_file = sys.argv[2]
    tokenization_error_rates_file = sys.argv[3]
    out_dir = sys.argv[4]
    seed = 20210424
    hyphenation_rate = 0.035
    powerlaw = "-power" in sys.argv or "-powerlaw" in sys.argv
    distribution = scipy.stats.powerlaw if powerlaw else scipy.stats.gamma
    fscale = 1.0 if "-fscale" in sys.argv else None
    zero = "-zero" in sys.argv

    random.seed(seed)

    ocr_error_rates = read_data(ocr_error_rates_file)
    tokenization_error_rates = read_data(tokenization_error_rates_file)
    ocr_p_zero = 0
    tokenization_p_zero = 0
    if zero:
        ocr_p_zero = len([rate for rate in ocr_error_rates if rate == 0]) / len(ocr_error_rates)
        ocr_error_rates = [rate for rate in ocr_error_rates if rate > 0]
        tokenization_p_zero = len([rate for rate in tokenization_error_rates if rate == 0]) \
            / len(tokenization_error_rates)
        tokenization_error_rates = [rate for rate in tokenization_error_rates if rate > 0]
    params_ocr = fit_distribution(ocr_error_rates, distribution, fscale)
    print("ocr", params_ocr)
    params_tokenization = fit_distribution(tokenization_error_rates, distribution, fscale)
    print("tokenization", params_tokenization)
    ocr_noise_inducer = ACLNoiseInducer(p=0, insertion_prob=0.2079, seed=seed)

    hyphenator = HyphenationIntroducer(hyphenation_rate)

    with open(out_dir + "/correct.txt", "w") as correct_file, open(out_dir + "/corrupt.txt", "w") as corrupt_file:
        for sequence in read_lines(in_file):
            print(sequence)
            spans = create_sequence_spans(sequence)
            misspelled_spans = []
            mistokenized_spans = []
            for span in spans:
                p_ocr = sample_distribution(params_ocr, distribution)
                ocr_noise_inducer.p = p_ocr
                p_space = sample_distribution(params_tokenization, distribution)
                misspelled = hyphenator.introduce_hyphens(span)
                if not (zero and flip_coin(random, ocr_p_zero)):
                    misspelled = ocr_noise_inducer.induce_noise(misspelled)
                if not (zero and flip_coin(random, tokenization_p_zero)):
                    mistokenized = corrupt_tokenization(misspelled, p_space)
                else:
                    mistokenized = misspelled
                misspelled_spans.append(misspelled)
                mistokenized_spans.append(mistokenized)
            misspelled = recombine_spans(misspelled_spans)
            mistokenized = recombine_spans(mistokenized_spans)
            print(misspelled)
            print(mistokenized)
            correct_file.write(misspelled + "\n")
            corrupt_file.write(mistokenized + "\n")
