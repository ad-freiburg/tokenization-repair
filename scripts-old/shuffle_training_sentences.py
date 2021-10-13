import random

import project
from src.helper.files import read_lines, write_lines
from src.settings import paths

if __name__ == "__main__":
    random.seed(42)
    print("reading...")
    lines = read_lines(paths.WIKI_TRAINING_SENTENCES)
    print("shuffling...")
    random.shuffle(lines)
    print("writing...")
    write_lines(paths.WIKI_TRAINING_SENTENCES_SHUFFLED, lines)
