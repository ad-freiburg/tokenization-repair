import argparse
from tqdm import tqdm

import project
from src.helper.pickle import dump_object
from src.helper.files import read_sequences


def main(args):
    print(f"counting chars in {args.text_file}...")
    frequencies = {}
    for line in tqdm(read_sequences(args.text_file)):
        for char in line:
            frequencies[char] = frequencies.get(char, 0) + 1
    print(f"dumping result to {args.output_file}...")
    dump_object(frequencies, args.output_file)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="text_file", help="input text file", required=True)
    parser.add_argument("-o", dest="output_file", help="output pickle file", required=True)
    args = parser.parse_args()
    main(args)
