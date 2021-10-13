import random

import project
from src.helper.files import read_lines


def read_typos(path):
    typos = []
    for line in read_lines(path):
        vals = line.split(" ")
        correct = vals[0]
        for i in range(1, len(vals), 2):
            misspelling = vals[i]
            frequency = int(vals[i + 1])
            for _ in range(frequency):
                typos.append((correct, misspelling))
    return typos


def write_typos(typos, path):
    typo_dict = {}
    for correct, misspelling in typos:
        if correct not in typo_dict:
            typo_dict[correct] = {}
        if misspelling not in typo_dict[correct]:
            typo_dict[correct][misspelling] = 1
        else:
            typo_dict[correct][misspelling] += 1
    with open(path, "w") as f:
        for correct in sorted(typo_dict):
            vals = [correct]
            for misspelling in sorted(typo_dict[correct]):
                vals.append(misspelling)
                vals.append(str(typo_dict[correct][misspelling]))
            f.write(" ".join(vals) + "\n")


if __name__ == "__main__":
    random.seed(20210515)
    directory = "/home/hertel/tokenization-repair-dumps/data/typos/"
    in_path = directory + "typo_corpus.txt"
    typos = read_typos(in_path)
    print(len(typos))
    random.shuffle(typos)
    n_test = len(typos) // 2
    test_typos = typos[:n_test]
    write_typos(test_typos, directory + "typos_test.txt")
    training_typos = typos[n_test:]
    write_typos(training_typos, directory + "typos_training.txt")
