#!/usr/bin/env python3
import argparse
import string
import time
import multiprocessing
import warnings
import random

import project
from src.noise.acl_noise_inducer import ACLNoiseInducer
from src.helper.files import read_lines
from src.helper.stochastic import flip_coin
warnings.filterwarnings("ignore")


def hyphenate(token):
    if len(token) < 2:
        return token
    positions = []
    for i in range(0, len(token) - 1):
        if token[i:(i + 2)].isalpha():
            positions.append(i)
    if len(positions) > 0:
        hyphen_pos = random.choice(positions) + 1
        token = token[:hyphen_pos] + "-" + token[hyphen_pos:]
    return token


def introduce_hyphens(sequence, p):
    tokens = sequence.split(" ")
    for i in range(len(tokens)):
        if flip_coin(random, p):
            tokens[i] = hyphenate(tokens[i])
    sequence = " ".join(tokens)
    return sequence


def add_typo(typo_dict, correct_spelling, misspelling, frequency=1):
    if correct_spelling not in typo_dict:
        typo_dict[correct_spelling] = []
    for _ in range(frequency):
        typo_dict[correct_spelling].append(misspelling)


def read_typos(typo_dict, typos_file: str):
    for line in read_lines(typos_file):
        vals = line.split(" ")
        correct = vals[0]
        for i in range(1, len(vals), 2):
            misspelling = vals[i]
            frequency = int(vals[i + 1])
            add_typo(typo_dict, correct, misspelling, frequency)


def corrupt_sequence_function(sequence):
    global noise_inducer
    if not noise_inducer.seed_is_set:
        seed = int(time.time() * 1000)
        random.seed(seed)
        noise_inducer.acl_inducer.rdm.seed(seed)
        noise_inducer.seed_is_set = True
    return noise_inducer.corrupt(sequence[:-1])


def corrupt_all(path, outpath, take=None, freq=10000):
    total = 108068848 if take is None else take
    last_out = ''
    with open(path, 'r') as src_file, open(outpath, 'w') as out_file:
        with multiprocessing.Pool(12) as pool:
            tic = time.time()
            # for idx, corrupt in enumerate(map(corrupt_sequence_function, src_file)):
            for idx, corrupt in enumerate(pool.imap(corrupt_sequence_function, src_file)):
                out_file.write(corrupt + '\n')  # print(corrupt, file=sys.stderr)
                if take is not None and idx >= take:
                    break
                if (idx % freq == freq - 1 or (take is not None and idx + 1 >= take)
                        or idx + 1 >= total):
                    toc = time.time()
                    rate = (toc - tic) / (idx + 1)
                    remain = int((total - idx - 1) * rate)
                    hour = '%.2d' % (remain // 3600)
                    remain = remain % 3600
                    minute = '%.2d' % (remain // 60)
                    seconds = '%.2d' % (remain % 60)
                    rate = '%.2f' % (rate * 1e6)
                    out = f'[{idx + 1}/{total}] {hour}:{minute}:{seconds} {rate} us/ex'

                    bs = len(last_out)
                    print('\b' * bs + ' ' * bs + '\b' * bs + out, end='',
                          flush=True)
                    last_out = out

        pool.join()
        print()


class CombinedNoiseInducer:
    def __init__(self, p_noise: float, p_hyphen: float, p_ocr_v_typo: float, p_ocr: float, p_typo: float,
                 p_random: float, p_insert: float,
                 typos_file: str, ocr_file: str):
        self.chars = string.ascii_lowercase
        self.p_noise = p_noise
        self.p_hyphen = p_hyphen
        self.p_ocr_v_typo = p_ocr_v_typo
        self.p_typo = p_typo
        self.p_random = p_random
        self.typos = {}
        read_typos(self.typos, typos_file=typos_file)
        self.acl_inducer = ACLNoiseInducer(p=p_ocr, insertion_prob=p_insert, seed=42, replacements_file=ocr_file)
        self.seed_is_set = False

    def random_char(self):
        return random.choice(self.chars)

    def random_error(self, token):
        pos = random.randint(0, len(token))
        type = random.randint(0, 3)
        if type == 0:
            return token[:pos] + self.random_char() + token[pos:]
        elif type == 1:
            return token[:pos] + token[(pos + 1):]
        elif type == 2:
            if pos == len(token):
                return token
            return token[:pos] + self.random_char() + token[(pos + 1):]
        else:
            if pos >= len(token) - 1:
                return token
            return token[:pos] + token[pos + 1] + token[pos] + token[(pos + 2):]

    def introduce_misspellings(self, sequence):
        tokens = sequence.split(" ")
        for i in range(len(tokens)):
            if random.random() < self.p_typo:
                if tokens[i] in self.typos and random.random() > self.p_random:
                    tokens[i] = random.choice(self.typos[tokens[i]])
                else:
                    tokens[i] = self.random_error(tokens[i])
        return " ".join(tokens)

    def introduce_ocr(self, sequence):
        return self.acl_inducer.induce_noise(sequence)

    def corrupt(self, sequence):
        if random.random() < self.p_noise:
            sequence = introduce_hyphens(sequence, self.p_hyphen)
            if random.random() < self.p_ocr_v_typo:
                sequence = self.introduce_ocr(sequence)
            else:
                sequence = self.introduce_misspellings(sequence)
        return sequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Efficient generation of ACL-like benchmark or training data "
        "by inducing similar OCR errors and additional spelling errors")
    parser.add_argument(
        '--src-path', help='source dataset path',
        default='/nfs/datasets/tokenization-repair/training_mixed.txt'
    )
    parser.add_argument(
        '--dest-path', help='destination dataset path',
        default='/nfs/students/mostafa-mohamed/tokenization-repair-paper/training_ocr.txt'
    )
    parser.add_argument(
        "--typos-file", help="File with word replacements",
        default="data/noise/typos_training.txt"
    )
    parser.add_argument(
        "--ocr-file", help="File with OCR replacement frequencies",
        default="data/noise/ocr_error_frequencies.ACL+ICDAR.weighted.tsv"
    )
    parser.add_argument(
        "--p-noise", type=float, default=1.0,
        help="Probability of inducing noise into a sequence."
    )
    parser.add_argument(
        "--p-hyphen", type=float, default=0.0114,
        help="When inducing noise, probability that a word gets hyphenated if hyphenation is possible."
    )
    parser.add_argument(
        "--p-ocr-v-typo", type=float, default=0.5,
        help="When inducing noise, probability of inducing OCR noise. "
             "Typo noise is induced with probability (1 - p_ocr_v_typo)."
    )
    parser.add_argument(
        "--p-ocr", type=float, default=0.1,
        help="When inducing OCR noise, probability that a word gets noised."
    )
    parser.add_argument(
        "--p-typo", type=float, default=0.1,
        help="When inducing typo noise, probability that a word gets noised."
    )
    parser.add_argument(
        "--p-random", type=float, default=0.5,
        help="When inducing a typo into a word, probability that a random typo is generated. "
             "A typo from the typo collection is sampled with probability (1 - p_random). "
             "When no typo for that word exists in the collection, always a random typo is generated."
    )
    parser.add_argument(
        "--p-insert", type=float, default=0.2079,
        help="When inducing an OCR error into a word, probability that the error is an insertion instead of a"
             "replacement or deletion."
    )
    parser.add_argument(
        "--seed", type=int, default=40,
        help="Random seed."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    noise_inducer = CombinedNoiseInducer(p_noise=args.p_noise, p_hyphen=args.p_hyphen, p_ocr_v_typo=args.p_ocr_v_typo,
                                         p_ocr=args.p_ocr, p_typo=args.p_typo, p_random=args.p_random,
                                         p_insert=args.p_insert,
                                         typos_file=args.typos_file, ocr_file=args.ocr_file)
    corrupt_all(args.src_path, args.dest_path)
