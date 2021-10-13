import argparse

import project
from src.noise.acl_noise_inducer import ACLNoiseInducer
from src.helper.time import time_diff, timestamp
from src.helper.files import read_lines


def main(args):
    inducer = ACLNoiseInducer(args.p, 0.2079, args.seed)

    if args.print_insertion_prob:
        error_dict = inducer.error_dict
        total_count = 0
        insertion_count = 0
        for correct in error_dict:
            for wrong, freq in error_dict[correct]:
                total_count += freq
                if correct == "":
                    insertion_count += freq
        insertion_prob = insertion_count / total_count
        print(len([e for e, f in error_dict[""] if f >= 0]), "insertions")
        print(f"{insertion_prob * 100:.2f}% char insertions ({insertion_count}/{total_count})")

    if args.runtime:
        sequence = "Tokenization Repair in the Presence of Spelling Errors"
        start_time = timestamp()
        corrupt_sequences = []
        for _ in range(100):
            corrupt_sequences.append(inducer.induce_noise(sequence))
        runtime = time_diff(start_time)
        for s in corrupt_sequences:
            print(s)
        print(runtime)
    elif args.input_file:
        out_file = open(args.output_file, "w") if args.output_file else None
        lines = read_lines(args.input_file)
        for line in lines:
            corrupt = inducer.induce_noise(line)
            print(corrupt)
            if out_file is not None:
                out_file.write(corrupt + "\n")
        if out_file is not None:
            out_file.close()
    else:
        while True:
            sequence = input("> ")
            for _ in range(100):
                corrupt = inducer.induce_noise(sequence)
                print(corrupt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true", default=False)
    parser.add_argument("-input_file", "-i", type=str)
    parser.add_argument("-output_file", "-o", type=str)
    parser.add_argument("-p", type=float, default=0.05)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("--print_insertion_prob", "-pib", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
