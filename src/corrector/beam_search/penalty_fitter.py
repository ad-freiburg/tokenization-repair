from typing import Dict, Tuple, List, Union, Optional

from enum import Enum
import numpy as np

from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.benchmark.benchmark import Benchmark, BenchmarkFiles
from src.sequence.transformation import space_corruption_positions


class Case(Enum):
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 1

    def __lt__(self, other):
        return self.value > other.value


def score(p: float):
    return -np.log(p)


def score_diff(p_good: float, p_bad: float):
    return score(p_bad) - score(p_good)


class PenaltyFitter:
    def __init__(self,
                 model_name: str,
                 labeling_model_name: Optional[str] = None,
                 n_sequences: int = -1):
        self.model = UnidirectionalLMEstimator()
        self.model.load(model_name)
        self.backward = self.model.specification.backward
        self.space_label = self.model.encoder.encode_char(' ')
        self.n_sequences = n_sequences
        if labeling_model_name is None:
            self.labeling_model = None
        else:
            self.labeling_model = BidirectionalLabelingEstimator()
            self.labeling_model.load(labeling_model_name)

    def space_and_nospace_probabilities(self,
                                        state: Dict,
                                        nospace_label: int,
                                        bidir_space_prob: Optional[float] = None):
        p_space = state["probabilities"][self.space_label]
        p_other = state["probabilities"][nospace_label]
        state_after_space = self.model.step(state, self.space_label, include_sequence=False)
        p_other_given_space = state_after_space["probabilities"][nospace_label]
        p_space_total = p_space * p_other_given_space
        if bidir_space_prob is not None:
            p_space_total = p_space_total * bidir_space_prob
            p_other = p_other * (1 - bidir_space_prob)
        return p_space_total, p_other

    @staticmethod
    def optimal_value(penalty_case_pairs, minimize_errors=False):
        penalty_case_pairs = sorted(penalty_case_pairs, reverse=True)
        total_true_positives = sum([1 for _, case in penalty_case_pairs if case == Case.TRUE_POSITIVE])
        penalty_case_pairs = [(penalty, case) for penalty, case in penalty_case_pairs if penalty > 0]
        penalties = [np.inf] + [t for t, _ in penalty_case_pairs]
        tp_vec = [0]
        fp_vec = [0]
        tps = 0
        fps = 0
        for _, case in penalty_case_pairs:
            if case == Case.TRUE_POSITIVE:
                tps += 1
            else:
                fps += 1
            tp_vec.append(tps)
            fp_vec.append(fps)
        tp_vec = np.asarray(tp_vec)
        fp_vec = np.asarray(fp_vec)
        if minimize_errors:
            fn_vec = total_true_positives - tp_vec
            errors = fn_vec + fp_vec
            best = int(np.argmin(errors))
            print("best errors=%i@%f (%i TP, %i FP, %i FN)" %
                  (errors[best], penalties[best], tp_vec[best], fp_vec[best], fn_vec[best]))
        else:
            precision = tp_vec / (tp_vec + fp_vec)
            recall = tp_vec / total_true_positives
            f1 = [2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0 for prec, rec in zip(precision, recall)]
            best = int(np.argmax(f1))
            print("best f1=%f@%f (precision=%f, recall=%f)" %
                  (f1[best], penalties[best], precision[best], recall[best]))
        return penalties[best]

    def fit(self,
            benchmarks: Union[List[Benchmark], Benchmark]) -> Dict[str, Tuple[float, float]]:
        """
        Assumes that all benchmarks share the same correct sequences.
        """
        if not isinstance(benchmarks, list):
            benchmarks = [benchmarks]
        benchmark_names = [benchmark.name for benchmark in benchmarks]
        benchmarks = {benchmark.name: benchmark for benchmark in benchmarks}
        insertions = {name: [] for name in benchmark_names}
        deletions = {name: [] for name in benchmark_names}

        correct_sequences = benchmarks[benchmark_names[0]].get_sequences(BenchmarkFiles.CORRECT)
        corrupt_sequences = {name: benchmarks[name].get_sequences(BenchmarkFiles.CORRUPT) for name in benchmark_names}
        for s_i, correct in enumerate(correct_sequences):
            if s_i == self.n_sequences:
                break
            print("sequence %i" % s_i)
            encoded = self.model.encoder.encode_sequence(correct)
            if self.model.specification.backward:
                correct = correct[::-1]
                encoded = encoded[::-1]
            insertion_positions = {}
            deletion_positions = {}
            for benchmark_name in benchmark_names:
                corrupt = corrupt_sequences[benchmark_name][s_i]
                if self.model.specification.backward:
                    corrupt = corrupt[::-1]
                ins_pos, del_pos = space_corruption_positions(correct, corrupt)
                insertion_positions[benchmark_name] = ins_pos
                deletion_positions[benchmark_name] = del_pos

            if self.labeling_model is not None:
                labeling_result = self.labeling_model.predict(correct.replace(' ', ''))
                labeling_space_probs = labeling_result["probabilities"]
            pos_in_no_space = 0

            state = self.model.initial_state()
            state = self.model.step(state, encoded[0])
            for i, char in enumerate(correct):
                other_label = encoded[i + 2] if char == ' ' else encoded[i + 1]
                char_before = correct[i - 1] if i > 0 else ''

                p_space_labeling = None if self.labeling_model is None else labeling_space_probs[pos_in_no_space]

                space_prob, other_prob = self.space_and_nospace_probabilities(state, other_label,
                                                                              bidir_space_prob=p_space_labeling)

                for benchmark_name in benchmark_names:
                    if i in insertion_positions[benchmark_name]:
                        penalty = score_diff(other_prob, space_prob)
                        deletions[benchmark_name].append((penalty, Case.TRUE_POSITIVE))
                    elif i in deletion_positions[benchmark_name]:
                        penalty = score_diff(space_prob, other_prob)
                        insertions[benchmark_name].append((penalty, Case.TRUE_POSITIVE))
                    elif char == ' ':
                        penalty = score_diff(other_prob, space_prob)
                        if penalty > 0:
                            deletions[benchmark_name].append((penalty, Case.FALSE_POSITIVE))
                    elif char_before != ' ':
                        penalty = score_diff(space_prob, other_prob)
                        if penalty > 0:
                            insertions[benchmark_name].append((penalty, Case.FALSE_POSITIVE))

                state = self.model.step(state, encoded[i + 1], include_sequence=False)
                if char != ' ':
                    pos_in_no_space += 1

        penalties = {}
        for benchmark_name in benchmark_names:
            print(benchmark_name, "insertion penalty:")
            insertion_penalty = -self.optimal_value(insertions[benchmark_name])
            print(benchmark_name, "deletion penalty:")
            deletion_penalty = -self.optimal_value(deletions[benchmark_name])
            penalties[benchmark_name] = (insertion_penalty, deletion_penalty)
        return penalties
