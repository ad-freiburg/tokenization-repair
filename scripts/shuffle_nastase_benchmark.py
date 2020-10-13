import random


if __name__ == "__main__":
    directory = "/home/hertel/tokenization-repair-dumps/nastase/acl-201302_word-resegmented/matched/"
    random.seed(13102020)

    n = 0
    for file in ["corrupt.txt", "nastase.txt"]:
        print(file)
        with open(directory + file) as f:
            lines = f.readlines()
        if n == 0:
            n = len(lines)
            indices = list(range(n))
            random.shuffle(indices)
        lines = [lines[i] for i in indices]
        with open(directory + file[:-4] + "_shuffled.txt", "w") as f:
            f.writelines(lines)
