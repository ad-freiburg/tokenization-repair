from typing import List

import sys
import numpy as np

import project
from src.helper.pickle import load_object
from src.settings import paths
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.sequence.functions import get_space_positions_in_merged
from development_probabilities import Case
from src.corrector.beam_search.penalty_fitter import Case as CaseLabel, PenaltyFitter
from src.corrector.beam_search.penalty_holder import PenaltyHolder


if __name__ == "__main__":
    model_name = sys.argv[1]
    benchmark_name = sys.argv[2]
    lookahead = int(sys.argv[3])
    cases_path = paths.CASES_FILE_NOISY if benchmark_name.startswith("0.1") else paths.CASES_FILE_CLEAN

    sequence_cases = load_object(cases_path)
    # sequence_cases: List[Case]

    insertion_cases = []
    deletion_cases = []

    benchmark = Benchmark(benchmark_name, Subset.TUNING)

    for s_i, sequence_pair in enumerate(benchmark.get_sequence_pairs(BenchmarkFiles.CORRUPT)):
        if s_i >= len(sequence_cases):
            break
        print(s_i)

        correct, corrupt = sequence_pair
        cases = sequence_cases[s_i]

        correct_pos = 0
        corrupt_pos = 0

        for i, case in enumerate(cases):
            true_space = correct[correct_pos] == ' ' if correct_pos < len(correct) else False
            input_space = corrupt[corrupt_pos] == ' ' if corrupt_pos < len(corrupt) else False

            space_score = np.sum(np.log([case.p_space] + [p for p in case.p_after_space[:lookahead]]))
            no_space_score = np.sum(np.log(case.p_after_no_space[:lookahead]))
            """print(correct[i] if i < len(correct) else "EOS",
                  true_space,
                  input_space,
                  space_score,
                  no_space_score)"""

            if input_space:
                # deletion
                score_diff = no_space_score - space_score
                label = CaseLabel.TRUE_POSITIVE if not true_space else CaseLabel.FALSE_POSITIVE
                #print("    deletion", score_diff, label)
                deletion_cases.append((score_diff, label))
            else:
                # insertion
                score_diff = space_score - no_space_score
                label = CaseLabel.TRUE_POSITIVE if true_space else CaseLabel.FALSE_POSITIVE
                #print("    insertion", score_diff, label)
                insertion_cases.append((score_diff, label))
            #if score_diff > 0:
            #    print("!!!!!")

            correct_pos += 1
            if true_space == input_space:
                corrupt_pos += 1
            elif input_space:
                corrupt_pos += 2

    insertion_penalty = -PenaltyFitter.optimal_value(insertion_cases)
    deletion_penalty = -PenaltyFitter.optimal_value(deletion_cases)

    print(insertion_penalty, deletion_penalty)

    holder = PenaltyHolder()
    penalty_name = model_name + "_lookahead%i" % lookahead
    holder.set(penalty_name, benchmark_name, insertion_penalty, deletion_penalty)
    print("saved.")
