import sys

from project import src
from src.settings import paths
from src.helper.files import read_lines
from src.arxiv.dataset import to_input_file


def to_paragraphs(lines):
    paragraphs = []
    for line in lines:
        if len(paragraphs) == 0 or len(line) == 0:
            paragraphs.append(line)
        else:
            if len(paragraphs[-1]) > 0:
                paragraphs[-1] += " "
            paragraphs[-1] += line
    paragraphs = [p for p in paragraphs if len(p) > 0]
    return paragraphs


def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def remove_diff_chars(paragraph: str) -> str:
    for char in (" ", "-"):
        paragraph = paragraph.replace(char, "")
    return paragraph


def get_input_sequence(s_true, s_in):
    corrected_input = ""
    t_i = s_i = 0
    while s_i < len(s_in):
        if s_true[t_i] == s_in[s_i]:
            corrected_input += s_true[t_i]
            t_i += 1
            s_i += 1
        elif s_true[t_i] == " ":
            t_i += 1
        elif s_in[s_i] == " ":
            corrected_input += " "
            s_i += 1
        elif s_true[t_i] == "-":
            corrected_input += "-"
            t_i += 1
        elif s_in[s_i:(s_i + 2)] == "- ":
            s_i += 2
        else:
            return None
    return corrected_input


if __name__ == "__main__":
    pdf_extractor = sys.argv[1]  # "pdftotext" # "pdfextract"

    files = read_lines(paths.ARXIV_DEVELOPMENT_FILES)

    base_path = paths.ARXIV_BASE_DIR
    out_path = "/home/hertel/tokenization-repair-dumps/benchmarks/%s/" % pdf_extractor

    n_matched = 0
    n_different = 0
    n_diff_spaces = 0

    for i, file in enumerate(files):
        true_path = base_path + "groundtruth/" + file
        input_path = base_path + pdf_extractor + "/" + to_input_file(file)

        true_lines = read_lines(true_path)
        true_lines = [clean_text(line) for line in true_lines]
        true_lines = [line for line  in true_lines if len(line) > 0]

        input_lines = read_lines(input_path)
        input_paragraphs = to_paragraphs(input_lines)
        input_paragraphs = [clean_text(p) for p in input_paragraphs]
        input_paragraphs = {remove_diff_chars(p): p for p in input_paragraphs}

        for line in true_lines:
            removed = remove_diff_chars(line)
            if removed in input_paragraphs:
                n_matched += 1
                input_sequence = get_input_sequence(line, input_paragraphs[removed])
                if input_sequence is None:
                    continue
                if input_sequence.replace(" ", "") != line.replace(" ", ""):
                    continue
                if line != input_paragraphs[removed]:
                    n_different += 1
                if line != input_sequence:
                    n_diff_spaces += 1
                    print(line)
                    print(input_sequence)
                    print()

    print(n_matched, "matched")
    print(n_different, "differing")
    print(n_diff_spaces, "differing at spaces")
