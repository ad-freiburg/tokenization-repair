import numpy as np
import os

import project
from src.benchmark.benchmark import BenchmarkFiles, Benchmark, Subset
from src.settings import paths
from src.evaluation.tolerant import tolerant_preprocess_sequences
from src.datasets.wikipedia import Wikipedia


APPROACHES = ["do nothing", "dynamic programming", "wordsegment", "BS fw wiki", "BS bidir wiki",
              "BS bidir wiki robust", "BS bidir combo robust"]

FILE_NAMES = {
    "do nothing": ["corrupt.txt"],
    "dynamic programming": ["bigrams.txt"],
    "wordsegment": ["wordsegment.txt"],
    # "google": ["google.txt"],
    "BS fw wiki": ["BS-fwd_wikipedia.SeqAccPenalties.txt", "beam_search.txt"],
    "BS bidir wiki": ["BS-bidir_wikipedia.SeqAccPenalties.txt", "beam_search_labeling_ce.txt"],
    "BS bidir wiki robust": ["BS-bidir_wikipedia-robust.SeqAccPenalties.txt",
                             "beam_search_labeling_robust_ce.txt"],
    "BS bidir combo robust": ["BS-bidir_combined-mixed-robust.SeqAccPenalties.txt",
                              "BS-bidir_combined-mixed-robust.txt"]
}


BENCHMARKS = [
    "0_0.1", "0_inf", "0.1_0.1", "0.1_inf", "nastase-big", "arxiv-910k"
]


def zero_one_sequence(a, b, originals=None):
    res = []
    for i, (aa, bb) in enumerate(zip(a, b)):
        if originals is not None:
            aa, _, bb = tolerant_preprocess_sequences(originals[i], aa, aa, bb)
        res.append(1 if aa == bb else 0)
    return res


if __name__ == "__main__":
    original_sequences = list(Wikipedia.test_sequences())
    for b in BENCHMARKS:
        print("**" + b + "**")
        benchmark = Benchmark(b, Subset.DEVELOPMENT if b == "nastase-big" else Subset.TEST)
        path = benchmark.get_results_directory()
        ground_truth_sequences = benchmark.get_sequences(BenchmarkFiles.CORRECT)
        with open(paths.DUMP_DIR + "zero_one_sequences/%s.txt" % b, "w") as f:
            for approach in APPROACHES:
                print(approach)
                if approach == "do nothing":
                    predicted_sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
                else:
                    file = None
                    for ff in FILE_NAMES[approach]:
                        if os.path.exists(benchmark.get_results_directory() + ff):
                            file = ff
                            break
                    predicted_sequences = benchmark.get_predicted_sequences(file)
                sequence = zero_one_sequence(ground_truth_sequences, predicted_sequences,
                                             original_sequences if b.startswith("0.1") else None)
                f.write(approach + "\n")
                f.write(''.join(str(x) for x in sequence) + "\n")
                print(np.mean(sequence))
        print()
