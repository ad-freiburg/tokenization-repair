import numpy as np

import project
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles


if __name__ == "__main__":
    benchmarks = [
        "ACL",
        "arXiv.OCR",
        "arXiv.pdftotext",
        "Wiki.no_spaces",
        "Wiki.spaces",
        "Wiki.typos.no_spaces",
        "Wiki.typos.spaces"
    ]

    approaches = [
        "corrupt.txt",
        "bigrams.txt",
        "wordsegment.txt",
        "google_deduced.txt",
        "BS-fwd.txt",
        "BS-fwd-OCR.txt",
        "BS-bid.txt",
        "BS-bid-OCR.txt"
    ]

    out_dir = "/home/hertel/tokenization-repair-dumps/data/zero_one_sequences/"

    for benchmark in benchmarks:
        print(benchmark)
        benchmark = Benchmark(benchmark, Subset.TEST)
        corrupt_sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        correct_sequences = benchmark.get_sequences(BenchmarkFiles.CORRECT)
        with open(out_dir + benchmark.name + ".txt", "w") as f:
            for approach in approaches:
                f.write(approach + "\n")
                predicted_sequences = corrupt_sequences if approach == "corrupt.txt" \
                    else benchmark.get_predicted_sequences(approach)
                zero_one_sequence = [1 if predicted == correct else 0
                                     for predicted, correct in zip(predicted_sequences, correct_sequences)]
                f.write("".join([str(i) for i in zero_one_sequence]) + "\n")
                print("", approach, np.mean(zero_one_sequence))
