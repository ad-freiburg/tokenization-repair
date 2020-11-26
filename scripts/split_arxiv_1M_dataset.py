from typing import Set

import sys
import os
import random


def get_text_files_recursive(directory) -> Set[str]:
    file_set = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".body.txt"):
                path = os.path.join(root, file)
                path = path[len(directory):]
                file_set.add(path)
    return file_set


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:\n"
              "  python3 scripts/split_arxiv_1M_dataset.py <dataset-dir> <benchmark-dir> <output-dir>\n"
              "dataset-dir: path to the ground truth files of the big dataset\n"
              "benchmark-dir: path to the ground truth files of the arxiv-benchmark with 12,000 files (those will be used as test set)\n"
              "output-dir: four files will be written here: training_files.txt, development_files.txt, tuning_files.txt and test_files.txt")
        exit(1)

    big_dataset_dir = sys.argv[1]  # "/nfs/datasets/arxiv/body-text/"  # "/mnt/datasets/arxiv/tokenization-repair-data/"
    benchmark_dir = sys.argv[2]  # "/nfs/datasets/arxiv/benchmark-data/benchmark/groundtruth/"  # "/home/hertel/tokenization-repair-dumps/claudius/groundtruth/"
    output_dir = sys.argv[3]

    random.seed(18112020)

    print("Reading dataset...")
    big_dataset_files = get_text_files_recursive(big_dataset_dir)
    print(len(big_dataset_files), "files")
    print("Reading benchmark...")
    benchmark_files = get_text_files_recursive(benchmark_dir)
    print(len(benchmark_files), "files")

    no_test_files = big_dataset_files.difference(benchmark_files)
    print(len(no_test_files), "not in benchmark")

    print("Shuffling...")
    no_test_files = list(no_test_files)
    random.shuffle(no_test_files)

    test_files = benchmark_files.intersection(benchmark_files)
    development_files = no_test_files[:10000]
    tuning_files = no_test_files[10000:20000]
    training_files = no_test_files[20000:]

    for files, file_name in [(test_files, "test_files.txt"),
                             (development_files, "development_files.txt"),
                             (tuning_files, "tuning_files.txt"),
                             (training_files, "training_files.txt")]:
        path = output_dir + file_name
        print("Writing %i files to %s" % (len(files), path))
        with open(path, "w") as f:
            for file in sorted(files):
                f.write(file + "\n")

    print("Done.")
