import random

if __name__ == "__main__":
    random.seed(13102020)
    directory = "/home/hertel/tokenization-repair-dumps/claudius/matched/"
    n = 0
    for file in ("corrupt.txt", "correct.txt"):
        with open(directory + file) as f:
            lines = f.readlines()
        if n == 0:
            n = len(lines)
            indices = list(range(n))
            random.shuffle(indices)
        lines = [lines[i] for i in indices]
        with open(directory + file[:-4] + "_shuffled.txt", "w") as f:
            f.writelines(lines)
