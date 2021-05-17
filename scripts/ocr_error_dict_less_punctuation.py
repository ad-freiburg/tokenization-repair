import project
from src.helper.files import read_lines


if __name__ == "__main__":
    path = "/home/hertel/tokenization-repair-dumps/data/dictionaries/"
    in_file = path + "ocr_error_frequencies.ACL+ICDAR.weighted.tsv"
    out_file = path + "ocr_error_frequencies.ACL+ICDAR.weighted.less_punctuation.tsv"
    with open(out_file, "w") as f:
        for line in read_lines(in_file):
            wrong, truth, frequency = line.split("\t")
            if len(wrong) == 0 or wrong.isalnum():
                frequency = str(int(frequency) * 10)
            f.write("\t".join((wrong, truth, frequency)) + "\n")
