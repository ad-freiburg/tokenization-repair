from typing import List

from os import listdir


def get_publication_year(file_name: str) -> int:
    year = int(file_name[1:3])
    if year < 21:
        year += 2000
    else:
        year += 1900
    return year


def get_lines(path: str) -> List[str]:
    with open(path, encoding="utf8") as f:
        lines = f.readlines()
    lines = [line[:-1] for line in lines]
    lines = [line for line in lines if len(line) > 0]
    return lines


def lines_remove_spaces(sequences: List[str]) -> List[str]:
    return [remove_spaces(s) for s in sequences]


def remove_spaces(sequence: str) -> str:
    sequence = sequence.replace(' ', '')
    return sequence


def reverse_engineer(input_text: str, truth_no_spaces: str) -> str:
    verbose = False # "-" in input_text and len(input_text) < 50
    if verbose:
        print(input_text)
        print(truth_no_spaces)
    transformed = ""
    i = 0
    j = 0
    while i < len(input_text) and j < len(truth_no_spaces):
        if verbose:
            print(i, j, transformed)
        if input_text[i] == "-" and truth_no_spaces[j] != "-":
            i += 1
            if i < len(input_text) and input_text[i] == " ":
                i += 1
        else:
            transformed += input_text[i]
            if input_text[i] != " ":
                j += 1
            i += 1
    return transformed


def preprocess_sequence(sequence: str):
    sequence = sequence.strip()
    return ' '.join(sequence.split())


if __name__ == "__main__":
    directory = "/home/hertel/tokenization-repair-dumps/nastase/acl-201302_word-resegmented/"
    raw_dir = directory + "raw/"
    re_segged_dir = directory + "re-segmented/"
    matched_dir = directory + "matched/"

    input_file = open(matched_dir + "corrupt.txt", "w")
    nastase_file = open(matched_dir + "nastase.txt", "w")

    files = listdir(raw_dir)
    files = sorted([file for file in files if get_publication_year(file) < 2005])

    # files = ["X98-1031.txt"]

    for file in files:
        print(file)
        lines = get_lines(raw_dir + file)
        lines_no_spaces = lines_remove_spaces(lines)
        re_segged = get_lines(re_segged_dir + file + ".re-segged")
        for prediction_line in re_segged:
            prediction_line_no_spaces = remove_spaces(prediction_line)
            for i, line_no_spaces in enumerate(lines_no_spaces):
                if prediction_line_no_spaces.startswith(line_no_spaces):
                    input_text = lines[i]
                    text_no_spaces = line_no_spaces
                    j = i + 1
                    while len(text_no_spaces) < len(prediction_line_no_spaces) and j < len(lines):
                        input_text += lines[j]
                        text_no_spaces += lines_no_spaces[j]
                        j += 1
                    if text_no_spaces.replace('-', '') == prediction_line_no_spaces.replace('-', ''):
                        #print(file, i)
                        transformed_input = reverse_engineer(input_text, prediction_line_no_spaces)
                        #print(transformed_input)
                        #print(prediction_line)
                        input_sequence = preprocess_sequence(transformed_input)
                        output_sequence = preprocess_sequence(prediction_line)
                        if input_sequence.replace(' ', '') != output_sequence.replace(' ', ''):
                            print("WARNING! Sequences differ not only at spaces.")
                            print(input_text)
                            print(input_sequence)
                            print(output_sequence)
                        else:
                            input_file.write(input_sequence + '\n')
                            nastase_file.write(output_sequence + '\n')
                            break

    input_file.close()
    nastase_file.close()
