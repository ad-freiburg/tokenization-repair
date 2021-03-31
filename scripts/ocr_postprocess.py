import sys

from project import src
from src.helper.files import read_lines


def get_space_positions(text):
    i = 0
    positions = []
    for char in text:
        if char == " ":
            positions.append(i)
        else:
            i += 1
    return positions


def get_tokens(text, input_spaces, tokenized_spaces):
    text = text.replace(" ", "")
    start = 0
    tokens = []
    space_removed = False
    for i in range(len(text) + 1):
        if i in tokenized_spaces or i == len(text):
            tokens.append((text[start:i], space_removed))
            start = i
            space_removed = False
        elif i in input_spaces:
            space_removed = True
    return tokens


def has_letter(token):
    result = False
    for char in token:
        if char.isalpha():
            return True
    return False


THRESHOLD = 0.6


def strip_token(token):
    if not has_letter(token):
        return "", token, ""
    leading_symbols = 0
    for i in range(len(token)):
        if token[i].isalpha():
            leading_symbols = i
            break
    trailing_symbols = 0
    for i in range(len(token)):
        if token[-i - 1].isalpha():
            trailing_symbols = i
            break
    prefix = token[:leading_symbols]
    suffix = "" if trailing_symbols == 0 else token[-trailing_symbols:]
    word = token[leading_symbols:(len(token) - trailing_symbols)]
    return prefix, word, suffix


def is_word(token):
    n_letters = len([char for char in token if char.isalpha()])
    return n_letters / len(token) >= THRESHOLD


def remove_symbols(word):
    return "".join(char for char in word if char.isalnum())


if __name__ == "__main__":
    input_file = sys.argv[1]
    tokenized_file = sys.argv[2]

    input_lines = read_lines(input_file)
    tokenized_lines = read_lines(tokenized_file)

    for sequence, tokenized in zip(input_lines, tokenized_lines):
        input_spaces = get_space_positions(sequence)
        tokenized_spaces = get_space_positions(tokenized)
        tokens = get_tokens(sequence, input_spaces, tokenized_spaces)
        postprocessed_tokens = []
        for token, space_removed in tokens:
            prefix, word, suffix = strip_token(token)
            if space_removed and is_word(word):
                removed = remove_symbols(word)
                postprocessed_tokens.append(prefix + removed + suffix)
            else:
                postprocessed_tokens.append(token)
        print(" ".join(postprocessed_tokens))
