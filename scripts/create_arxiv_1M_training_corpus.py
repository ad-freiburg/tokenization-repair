import sys
import random
import os


def read_lines(path):
    with open(path, encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
    return lines


def preprocess_line(line):
    return " ".join(line.split()).strip()


def preprocess_lines(lines):
    return [preprocess_line(line) for line in lines]


def filter_lines(lines):
    return [line for line in lines if len(line) > 0 and line != "[formula]"]


if __name__ == "__main__":
    # python3 scripts/create_arxiv-1M_training_corpus.py /nfs/datasets/arxiv/tokenization-repair-data/ /local/data/hertelm/tokenization-repair-dumps/arxiv_1M/training_files.txt /local/data/hertelm/tokenization-repair-dumps/arxiv_1M/training.txt

    base_dir = sys.argv[1]
    file_list = sys.argv[2]
    out_file = sys.argv[3]

    if not base_dir.endswith("/"):
        base_dir += "/"

    files = read_lines(file_list)

    paragraphs = []

    print("Reading...")
    for i, file in enumerate(files):
        path = base_dir + file
        if os.path.exists(path):
            try:
                lines = read_lines(path)
                lines = preprocess_lines(lines)
                lines = filter_lines(lines)
                paragraphs.extend(lines)
            except UnicodeEncodeError:
                print("Unicode error in file:", file)

        if i % 100 == 0 or i + 1 == len(files):
            print("%i/%i files (%.2f%%): %i paragraphs" % (i, len(files), i / len(files) * 100, len(paragraphs)), end="\r")
    print("\nDone reading.")

    print("Shuffling...")
    random.shuffle(paragraphs)

    print("Writing...")
    with open(out_file, "w", encoding="utf8", errors="replace") as f:
        for p in paragraphs:
            f.write(p + "\n")
