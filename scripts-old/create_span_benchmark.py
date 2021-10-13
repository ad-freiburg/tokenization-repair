import sys
import random

import project
from src.helper.files import read_lines
from src.noise.acl_noise_inducer import ACLNoiseInducer
from src.helper.stochastic import flip_coin
from create_gamma_benchmark import HyphenationIntroducer


def is_punctuation(char):
    return not char.isalnum()


def corrupt_tokenization(text: str, p_insertion: float, p_deletion: float):
    if len(text) < 2:
        return text
    corrupt = text[0]
    for i in range(1, len(text)):
        if text[i] == " ":
            if not flip_coin(random, p_deletion):
                corrupt += " "
        else:
            if (i + 1 == len(text) or text[i + 1] != " ") and flip_coin(random, p_insertion):
                corrupt += " "
            corrupt += text[i]
    return corrupt


def unify_spaces(text):
    tokens = text.split(" ")
    tokens = [t for t in tokens if len(t) > 0]
    return " ".join(tokens)


if __name__ == "__main__":
    random.seed(11052021)

    directory = sys.argv[1]
    ocr_noise_inducer = ACLNoiseInducer(p=0, insertion_prob=0.201, seed=20210511)

    hyphenator = HyphenationIntroducer(p_hyphen=0.035)

    span_error_rates = []
    for line in read_lines("char_error_distributions/spans.txt"):
        vals = line.split("\t")
        span_error_rates.append((int(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])))

    p_erroneous_span = 910 / (16229 - 1429 + 910)

    corrupt_file = open(directory + "/corrupt.txt", "w")
    correct_file = open(directory + "/correct.txt", "w")

    for line in read_lines(directory + "/spelling.txt"):
        print(line)
        line = hyphenator.introduce_hyphens(line)
        print(line)
        tokens = line.split(" ")
        ground_truth = ""
        corrupt = ""
        i = 0
        while i < len(tokens):
            if flip_coin(random, p_erroneous_span):
                span_data = random.choice(span_error_rates)
                span_length, p_insertion, p_deletion, p_ocr = span_data
                print(span_data)
                span_tokens = tokens[i:(i + span_length)]
                span_text = " ".join(span_tokens)
                if p_ocr > 0:
                    ocr_noise_inducer.p = p_ocr
                    span_text = ocr_noise_inducer.induce_noise(span_text)
                ground_truth += " " + span_text
                corrupt += " " + corrupt_tokenization(span_text, p_insertion, p_deletion)
                i += span_length
            else:
                ground_truth += " " + tokens[i]
                corrupt += " " + tokens[i]
                i += 1
        ground_truth = unify_spaces(ground_truth)
        corrupt = unify_spaces(corrupt)
        print(ground_truth)
        print(corrupt)
        corrupt_file.write(corrupt + "\n")
        correct_file.write(ground_truth + "\n")

    corrupt_file.close()
    correct_file.close()
