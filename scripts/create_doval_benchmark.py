from typing import Tuple

import sys

import project
from src.helper.files import read_lines


CORRECT_MARKER = "Acierto!"
FALSE_MARKER = "Fallo!"


def extract_sequences(result: str, input: str) -> Tuple[str, str]:
    result = result[(len(FALSE_MARKER) + 1):]
    split = result.split(' ')
    split = [val for val in split if len(val) > 0]
    split_pt = 0
    acc_len = 0
    while acc_len < len(input):
        acc_len += len(split[split_pt])
        split_pt += 1
    predicted = ' '.join(split[:split_pt])
    correct = ' '.join(split[(split_pt + 1):])
    return predicted, correct


if __name__ == "__main__":
    in_file = sys.argv[1]
    lines_per_case = int(sys.argv[2])
    mode = sys.argv[3]  # correct, corrupt, predicted
    lines = read_lines(in_file)
    lines = [line[:-1] if line[-1] == '\t' else line for line in lines]
    n_cases = len(lines) // lines_per_case
    for i in range(n_cases):
        if lines_per_case == 3:
            input = lines[lines_per_case * i]
            result = lines[lines_per_case * i + 1]
        else:
            result = lines[lines_per_case * i]
            input = ''.join(result.split(" vs ")[0].split(' ')[1:])
        if result.startswith(CORRECT_MARKER):
            correct = predicted = result[(len(CORRECT_MARKER) + 1):]
        elif result.startswith(FALSE_MARKER):
            predicted, correct = extract_sequences(result,  input)
        else:
            raise Exception()
        if mode == "correct":
            print(correct)
        elif mode == "corrupt":
            print(input)
        elif mode == "predicted":
            print(predicted)
        else:
            raise Exception("Unknown mode '%s'." % str(mode))
