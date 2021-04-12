import project
from src.helper.files import read_lines


if __name__ == "__main__":
    folder = "/home/hertel/tokenization-repair-dumps/data/spelling/ACL/development/"
    out_folder = "acl_error_distribution/"
    n = 320

    n_error_free = 0

    for s_i, (correct, corrupt) in enumerate(zip(read_lines(folder + "spelling.txt"),
                                                 read_lines(folder + "corrupt.txt"))):
        if correct == corrupt:
            print(correct)
            n_error_free += 1
        if s_i + 1 == n:
            break

    percentage = n_error_free / n * 100
    print(f"{n_error_free} / {n} error free ({percentage}%)")