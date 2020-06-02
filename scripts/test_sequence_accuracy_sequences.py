import numpy as np

import project
from src.benchmark.benchmark import BenchmarkFiles, ERROR_PROBABILITIES, NOISE_LEVELS, get_benchmark, Subset
from src.helper.files import get_files
from src.settings import paths
from src.evaluation.tolerant import tolerant_preprocess_sequences
from src.datasets.wikipedia import Wikipedia


APPROACH_NAMES = {
    "greedy.txt": "greedy",
    "bigrams.txt": "bigrams",
    "wordsegment.txt": "wordsegment",
    "beam_search.txt": "BS fw",
    "beam_search_bwd.txt": "BS bw",
    "beam_search_robust.txt": "BS fw robust",
    "beam_search_bwd_robust.txt": "BS bw robust",
    "labeling_ce.txt": "bidirectional",
    "labeling_noisy_ce.txt": "bidirectional robust",
    "two_pass.txt": "two-pass",
    "two_pass_robust.txt": "two-pass robust",
    "beam_search_labeling_ce.txt": "BS fw+bi",
    "beam_search_labeling_robust_ce.txt": "BS fw+bi robust",
    "spelling_beam_search_segmentation.txt": "spell check BS"
}


def zero_one_sequence(a, b, originals):
    res = []
    for aa, bb, original in zip(a, b, originals):
        aa, _, bb = tolerant_preprocess_sequences(original, aa, aa, bb)
        res.append(1 if aa == bb else 0)
    return res


if __name__ == "__main__":
    original_sequences = list(Wikipedia.test_sequences())
    for n in NOISE_LEVELS:
        typo_str = "no typos" if n == 0 else "10% typos"
        for p in ERROR_PROBABILITIES:
            space_str = "no spaces" if p == np.inf else ("%.1f expected token errors" % p)
            print("%s, %s" % (typo_str, space_str))
            benchmark = get_benchmark(n, p, Subset.TEST)
            path = benchmark.get_results_directory()
            files = get_files(path)
            ground_truth_sequences = benchmark.get_sequences(BenchmarkFiles.CORRECT)
            with open(paths.DUMP_DIR + "zero_one_sequences/%s.txt" % benchmark.name, "w") as f:
                f.write("%s, %s\n" % (typo_str, space_str))
                for file in sorted(files):
                    if file in APPROACH_NAMES:
                        approach_name = APPROACH_NAMES[file]
                        print(approach_name)
                        predicted_sequences = benchmark.get_predicted_sequences(file)
                        sequence = zero_one_sequence(ground_truth_sequences, predicted_sequences, original_sequences)
                        f.write(approach_name + "\n")
                        f.write(''.join(str(x) for x in sequence) + "\n")
                        print(np.mean(sequence))
