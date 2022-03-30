#!/usr/bin/env python3
import argparse
import string
import time
import multiprocessing
import warnings
import random

import project
from src.noise.acl_noise_inducer import ACLNoiseInducer
from create_wikipedia_benchmark import TypoNoiseInducer
from introduce_hyphens import introduce_hyphens
warnings.filterwarnings("ignore")


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
    def __init__(self, typos_file, ocr_file):
        self.chars = string.ascii_lowercase
        self.p_hyphen = 0.0114
        self.p_typo = 0.1
        self.typos = TypoNoiseInducer(p=0.1, seed=41, test=False, typos_file=typos_file).typos
        self.acl_inducer = ACLNoiseInducer(p=0.1, insertion_prob=0.2079, seed=42, replacements_file=ocr_file)
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
                if tokens[i] in self.typos and random.random() < 0.5:
                    tokens[i] = random.choice(self.typos[tokens[i]])
                else:
                    tokens[i] = self.random_error(tokens[i])
        return " ".join(tokens)

    def introduce_ocr(self, sequence):
        return self.acl_inducer.induce_noise(sequence)

    def corrupt(self, sequence):
        sequence = introduce_hyphens(sequence, self.p_hyphen)
        if random.random() < 0.5:
            sequence = self.introduce_misspellings(sequence)
        else:
            sequence = self.introduce_ocr(sequence)
        return sequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Efficient generation of ACL-like benchmark by inducing similar OCR errors")
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
    args = parser.parse_args()

    random.seed(40)
    noise_inducer = CombinedNoiseInducer(args.typos_file, args.ocr_file)
    corrupt_all(args.src_path, args.dest_path)
