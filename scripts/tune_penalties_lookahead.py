from typing import List

import project

from src.interactive.parameters import ParameterGetter, Parameter


params = [Parameter("model_name", "-m", "str"),
          Parameter("labeling", "-labeling", "str"),
          Parameter("benchmark", "-b", "str"),
          Parameter("sequences", "-seq", "str"),
          Parameter("lookahead", "-l", "int")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import numpy as np

from src.helper.pickle import load_object
from src.helper.data_structures import izip
from src.settings import paths
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.corrector.beam_search.penalty_tuning import Case
from src.corrector.beam_search.penalty_fitter import Case as CaseLabel, PenaltyFitter
from src.corrector.beam_search.penalty_holder import PenaltyHolder


if __name__ == "__main__":
    model_name = parameters["model_name"]
    benchmark_name = parameters["benchmark"]
    lookahead = parameters["lookahead"]
    sequence_file = parameters["sequences"]
    cases_path = paths.CASES_FILE_NOISY if benchmark_name.startswith("0.1") else paths.CASES_FILE_CLEAN

    sequence_cases = load_object(cases_path)[model_name]
    # sequence_cases: List[Case]

    labeling = parameters["labeling"] != "0"
    if labeling:
        from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
        labeling_model = BidirectionalLabelingEstimator()
        labeling_model.load(parameters["labeling"])
    else:
        labeling_model = None

    insertion_cases = []
    deletion_cases = []

    benchmark = Benchmark(benchmark_name, Subset.TUNING)
    correct_sequences = benchmark.get_sequences(BenchmarkFiles.CORRECT)

    two_pass = sequence_file != "corrupt"
    if two_pass:
        input_sequences = benchmark.get_predicted_sequences(sequence_file)
    else:
        input_sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)

    n_sequences = 0
    for s_i, correct, corrupt in izip(correct_sequences, input_sequences):
        n_sequences += 1
        if s_i >= len(sequence_cases):
            break
        print(s_i)

        cases = sequence_cases[s_i]
        if model_name.startswith("bwd"):
            cases = cases[1:]

        labeling_space_probs = labeling_model.predict(correct.replace(' ', ''))["probabilities"] if labeling \
            else None

        correct_pos = 0
        corrupt_pos = 0
        labeling_pos = 0

        for i, case in enumerate(cases):
            true_space = correct[correct_pos] == ' ' if correct_pos < len(correct) else False
            input_space = corrupt[corrupt_pos] == ' ' if corrupt_pos < len(corrupt) else False

            space_score = np.sum(np.log([case.p_space] + [p for p in case.p_after_space[:lookahead]]))
            no_space_score = np.sum(np.log(case.p_after_no_space[:lookahead]))

            if labeling:
                p_space_labeling = labeling_space_probs[labeling_pos]
                #print(labeling_pos, p_space_labeling)
                space_score += np.log(p_space_labeling)
                no_space_score += np.log(1 - p_space_labeling)

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
            if not true_space:
                labeling_pos += 1
    print("%i sequences" % n_sequences)

    print("insertion:")
    insertion_penalty = -PenaltyFitter.optimal_value(insertion_cases, minimize_errors=two_pass)
    print("deletion:")
    deletion_penalty = -PenaltyFitter.optimal_value(deletion_cases, minimize_errors=two_pass)

    print(insertion_penalty, deletion_penalty)

    holder = PenaltyHolder(two_pass=two_pass)
    penalty_name = model_name
    if labeling:
        penalty_name += "_%s" % parameters["labeling"]
    penalty_name += "_lookahead%i" % lookahead
    holder.set(penalty_name, benchmark_name, insertion_penalty, deletion_penalty)
    print("saved.")
