import random
import sys
import os

import project
from src.helper.files import read_lines
from src.settings.paths import DUMP_DIR
from src.helper.stochastic import flip_coin
from src.noise.token_corruptor import TokenCorruptor
from src.settings import constants


TYPO_DIR = DUMP_DIR + "typos/"


def add_typo(typo_dict, correct_spelling, misspelling, frequency=1):
    if correct_spelling not in typo_dict:
        typo_dict[correct_spelling] = []
    for _ in range(frequency):
        typo_dict[correct_spelling].append(misspelling)


def read_typos(typo_dict, test: bool):
    file_name = "typos_test.txt" if test else "typos_training.txt"
    for line in read_lines(TYPO_DIR + file_name):
        vals = line.split(" ")
        correct = vals[0]
        for i in range(1, len(vals), 2):
            misspelling = vals[i]
            frequency = int(vals[i + 1])
            add_typo(typo_dict, correct, misspelling, frequency)


class TypoNoiseInducer:
    def __init__(self, p: float, seed: int, test: bool):
        self.p = p
        self.typos = {}
        read_typos(self.typos, test)
        self.rdm = random.Random(seed)

    def corrupt(self, sequence: str) -> str:
        tokens = sequence.split(" ")
        for i in range(len(tokens)):
            if tokens[i] in self.typos:
                if flip_coin(self.rdm, self.p):
                    tokens[i] = self.rdm.choice(self.typos[tokens[i]])
        sequence = " ".join(tokens)
        return sequence


class SpaceRemover:
    def corrupt(self, sequence: str):
        return sequence.replace(" ", "")


if __name__ == "__main__":
    typos = sys.argv[1] == "1"
    spaces = sys.argv[2] == "1"

    in_dir = DUMP_DIR + "wiki/"

    benchmark_name = "Wiki"

    if typos:
        benchmark_name += ".typos-split"

    if spaces:
        corruptor = TokenCorruptor(p=0.1,
                                   positions_per_token=constants.POSITIONS_PER_TOKEN,
                                   token_pairs_per_token=constants.TOKEN_PAIRS_PER_TOKEN,
                                   seed=13052021)
        benchmark_name += ".spaces"
    else:
        corruptor = SpaceRemover()
        benchmark_name += ".no_spaces"

    out_dir = "/home/hertel/tokenization-repair-dumps/data/benchmarks/" + benchmark_name + "/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    i = 0
    for set in ("tuning", "development", "test"):
        if typos:
            test = set == "test"
            typo_inducer = TypoNoiseInducer(0.1, seed=20210513 + i, test=test)
            i += 1
        subdir = out_dir + set + "/"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        with open(subdir + "correct.txt", "w") as correct_file, open(subdir + "corrupt.txt", "w") as corrupt_file:
            sequences = read_lines(in_dir + set + ".txt")
            for correct in sequences:
                if typos:
                    correct = typo_inducer.corrupt(correct)
                corrupt = corruptor.corrupt(correct)
                print(corrupt)
                correct_file.write(correct + "\n")
                corrupt_file.write(corrupt + "\n")
