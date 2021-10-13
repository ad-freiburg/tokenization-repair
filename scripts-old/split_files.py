from os import path, listdir
import random


if __name__ == "__main__":
    random.seed(20201108)

    in_path = "/home/hertel/tokenization-repair-dumps/claudius/groundtruth-with-normalized-formulas/"
    n_test = 1000

    files = []

    for folder in sorted(listdir(in_path)):
        if path.isdir(in_path + folder):
            print(folder)
            for file in listdir(in_path + folder):
                if file.endswith(".txt"):
                    files.append(folder + "/" + file)

    random.shuffle(files)

    for subi, subset in enumerate(("development", "test", "training")):
        begin = subi * n_test
        end = begin + n_test if subi < 2 else -1
        path = in_path + subset + ".txt"
        with open(path, "w") as f:
            for file in sorted(files[begin:end]):
                f.write(file + "\n")
        print(path)
