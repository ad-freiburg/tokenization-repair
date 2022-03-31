import argparse

import project
from src.helper.pickle import load_object
from src.helper.data_structures import sort_dict_by_value


def main(args):
    frequencies = load_object(args.frequencies_dict)
    for elem in sort_dict_by_value(frequencies):
        print(elem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frequencies_dict")
    args = parser.parse_args()
    main(args)
