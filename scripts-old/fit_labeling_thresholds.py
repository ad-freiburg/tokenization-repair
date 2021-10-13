import sys
from enum import Enum
import numpy as np

import project

from src.corrector.labeling.labeling_corrector import LabelingCorrector
from src.corrector.threshold_holder import ThresholdHolder, FittingMethod
from src.benchmark.benchmark import Benchmark, BenchmarkFiles, ERROR_PROBABILITIES, get_benchmark_name, Subset
from src.sequence.functions import remove_spaces, get_space_positions_in_merged


class Case(Enum):
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 1

    def __lt__(self, other):
        return self == Case.TRUE_POSITIVE and other == Case.FALSE_POSITIVE


def transformed_space_positions(sequence: str):
    return {pos - 1 for pos in get_space_positions_in_merged(sequence)}


def optimal_value(cases):
    best_f1 = 0
    best_t = np.inf
    n_positive = len([case for _, case in cases if case == Case.TRUE_POSITIVE])
    tp = 0
    fp = 0
    fn = n_positive
    for p, case in sorted(cases, reverse=True):
        if case == Case.TRUE_POSITIVE:
            tp += 1
            fn -= 1
        else:
            fp += 1
        prec = (tp / (tp + fp)) if tp + fp > 0 else 0
        rec = tp / n_positive
        f1 = (2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_t = p
    return best_f1, best_t


if __name__ == "__main__":
    model_name = sys.argv[1]
    noise = float(sys.argv[2])

    corrector = LabelingCorrector(model_name, 0, 0)

    benchmark_names = [get_benchmark_name(noise, p) for p in ERROR_PROBABILITIES]
    benchmarks = {name: Benchmark(name, Subset.TUNING) for name in benchmark_names}
    input_sequences = {name: benchmarks[name].get_sequences(BenchmarkFiles.CORRUPT) for name in benchmark_names}
    correct_sequences = benchmarks[benchmark_names[0]].get_sequences(BenchmarkFiles.CORRECT)

    cases = {name: {"deletions": [], "insertions": []} for name in benchmark_names}

    for s_i, correct in enumerate(correct_sequences):
        print(s_i, correct)
        merged = remove_spaces(correct)
        space_probabilities = corrector.get_space_probabilities(merged)
        correct_spaces = transformed_space_positions(correct)
        for name in benchmark_names:
            input_sequence = input_sequences[name][s_i]
            input_spaces = transformed_space_positions(input_sequence)
            for pos, p in enumerate(space_probabilities):
                if pos in input_spaces:
                    # deletion
                    del_p = 1 - p
                    if pos in correct_spaces:
                        case = Case.FALSE_POSITIVE
                    else:
                        case = Case.TRUE_POSITIVE
                    cases[name]["deletions"].append((del_p, case))
                else:
                    # insertion
                    if pos in correct_spaces:
                        case = Case.TRUE_POSITIVE
                    else:
                        case = Case.FALSE_POSITIVE
                    cases[name]["insertions"].append((p, case))

    holder = ThresholdHolder(fitting_method=FittingMethod.LABELING)
    for name in benchmark_names:
        insertion_f1, insertion_threshold = optimal_value(cases[name]["insertions"])
        deletion_f1, deletion_threshold = optimal_value(cases[name]["deletions"])
        print(name)
        print("insertion f1=%.4f@%.4f" % (insertion_f1, insertion_threshold))
        print("deletion f1=%.4f@%.4f" % (deletion_f1, deletion_threshold))
        holder.set_insertion_threshold(model_name, threshold=insertion_threshold, noise_type=name)
        holder.set_deletion_threshold(model_name, threshold=deletion_threshold, noise_type=name)
    holder.save()
