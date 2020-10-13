from typing import List, Tuple

from os import listdir, path


def read_lines(path: str) -> List[str]:
    with open(path) as f:
        lines = f.readlines()
    lines = [line[:-1] for line in lines]
    lines = [" ".join(line.split()) for line in lines]
    lines = [line for line in lines if len(line) > 0]
    return lines


def match_lines(truth_path: str, input_path: str) -> List[Tuple[str, str]]:
    true_lines = read_lines(truth_path)
    true_dict = {line.replace(" ", ""): line for line in true_lines}
    pairs = []
    for input_line in read_lines(input_path):
        input_no_spaces = input_line.replace(" ", "")
        if input_no_spaces in true_dict:
            pairs.append((input_line, true_dict[input_no_spaces]))
    return pairs


if __name__ == "__main__":
    base_dir = "/home/hertel/tokenization-repair-dumps/claudius/"
    ground_truth_dir = base_dir + "groundtruth/"
    inputs_dir = base_dir + "PDFExtract/"
    matched_dir = base_dir + "matched/"
    truth_file_suffix = "body.txt"
    input_file_suffix = "final.txt"

    out_file_inputs = open(matched_dir + "corrupt.txt", "w")
    out_file_truth = open(matched_dir + "correct.txt", "w")

    folders = listdir(ground_truth_dir)

    matched_total = 0
    different_total = 0

    for folder in folders:
        files = listdir(ground_truth_dir + folder)

        for file in files:
            gt_path = ground_truth_dir + folder + "/" + file
            input_path = inputs_dir + folder + "/" + file[:-len(truth_file_suffix)] + input_file_suffix

            if path.exists(input_path):
                pairs = match_lines(gt_path, input_path)
                n_different = sum([1 if input != true else 0 for input, true in pairs])
                matched_total += len(pairs)
                different_total += n_different
                for input, true in pairs:
                    out_file_inputs.write(input + '\n')
                    out_file_truth.write(true + '\n')

        print(folder, matched_total, different_total)

    out_file_inputs.close()
    out_file_truth.close()
