import sys
import random

import project
from src.helper.files import get_files, read_file


def get_year(file_name: str) -> int:
    year_str = file_name[1:3]
    year = int(year_str)
    if year > 20:
        year += 1900
    else:
        year += 2000
    return year


if __name__ == "__main__":
    n = 100
    n_lines = 10

    acl_dir = "/home/hertel/tokenization-repair-dumps/nastase/acl-201302_word-resegmented/"
    if sys.argv[1] == "raw":
        acl_dir += "raw/"
    else:
        acl_dir += "re-segmented/"
    
    files = sorted(get_files(acl_dir))

    files = [file for file in files if get_year(file) < 2005]

    random.seed(42)
    random.shuffle(files)
    files = files[:n]

    benchmark_str = ""

    for f_i, file in enumerate(files):
        benchmark_str += "# %i: %s #\n" % (f_i + 1, file)
        content = read_file(acl_dir + file)
        if "-lines" in sys.argv:
            lines = content.split("\n")
            start_line = random.randint(0, max(0, len(lines) - n_lines - 1))
            content = "\n".join(lines[start_line:(start_line + n_lines)])
            if f_i + 1 < len(files):
                content += "\n"
        benchmark_str += content

    print(benchmark_str)
