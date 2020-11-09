import random

from project import src
from src.helper.files import read_lines


if __name__ == "__main__":
    random.seed(20201108)

    files_file = "/home/hertel/tokenization-repair-dumps/claudius/groundtruth-with-normalized-formulas/training.txt"
    base_dir = "/".join(files_file.split("/")[:-1]) + "/"
    print(base_dir)
    out_file = base_dir + "training_paragraphs.txt"

    files = read_lines(files_file)
    paragraphs = []

    print("reading...")

    for file in files:
        lines = read_lines(base_dir + file)
        lines = [" ".join(line.split()).strip() for line in lines]
        lines = [line for line in lines if len(line) > 0 and line != "[formula]"]
        paragraphs.extend(lines)

    print("shuffling...")

    random.shuffle(paragraphs)

    print("writing...")

    with open(out_file, "w") as f:
        for paragraph in paragraphs:
            f.write(paragraph + "\n")
