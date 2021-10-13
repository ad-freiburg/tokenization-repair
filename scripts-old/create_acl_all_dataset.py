import sys
import random
import shutil

import project
from src.helper.files import get_files, read_lines, write_lines
from src.settings import paths, symbols
from src.helper.pickle import dump_object


if __name__ == "__main__":
    random.seed(42)

    step = sys.argv[1]

    if step == "split":
        path = "/home/hertel/tokenization-repair-dumps/nastase/acl-201302_word-resegmented/raw/"
        files = sorted(get_files(path))
        print(len(files), "files")

        random.shuffle(files)
        n_test = 100

        out_path = "/home/hertel/tokenization-repair-dumps/acl_corpus/"
        for i, filename in enumerate(files):
            print(filename)
            if i < n_test:
                subdir = "development/"
            elif i < 2 * n_test:
                subdir = "test/"
            else:
                subdir = "training/"
            shutil.copy(path + filename, out_path + subdir + filename)

    elif step == "lines":
        for split in ["training", "development", "test"]:
            path = paths.ACL_CORPUS_DIR + split + "/"
            lines = []
            for filename in sorted(get_files(path)):
                lines.extend(read_lines(path + filename))
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line) > 0]
            lines = [' '.join(line.split()) for line in lines]
            lines = [line for line in lines if sum(1 if c == "?" else 0 for c in line) < 4]  # remove lines with many ?s
            print(len(lines), "lines")
            write_lines(paths.ACL_CORPUS_DIR + split + ".txt", lines)
            random.shuffle(lines)
            write_lines(paths.ACL_CORPUS_DIR + split + "_shuffled.txt", lines)

    elif step == "dict":
        char_frequencies = {}
        for line in read_lines(paths.ACL_CORPUS_TRAINING_FILE):
            for char in line:
                if char not in char_frequencies:
                    char_frequencies[char] = 1
                else:
                    char_frequencies[char] += 1
        print("== FREQUENCIES ==")
        for char in sorted(char_frequencies):
            print(char, char_frequencies[char])
        print("== ENCODER DICT ==")
        encoder_dict = {}
        for char in sorted(char_frequencies):
            if char_frequencies[char] > 10:
                encoder_dict[char] = len(encoder_dict)
        encoder_dict[symbols.SOS] = len(encoder_dict)
        encoder_dict[symbols.EOS] = len(encoder_dict)
        encoder_dict[symbols.UNKNOWN] = len(encoder_dict)
        print(encoder_dict)
        dump_object(encoder_dict, paths.ACL_ENCODER_DICT)
