import sys

import project
from src.settings import paths
from src.helper.pickle import load_object
from src.helper.data_structures import sort_dict_by_value


if __name__ == "__main__":
    char_dict = load_object(paths.CHARACTER_FREQUENCY_DICT)

    for i, (char, frequency) in enumerate(sort_dict_by_value(char_dict)):
        print(i + 1, char, frequency)
