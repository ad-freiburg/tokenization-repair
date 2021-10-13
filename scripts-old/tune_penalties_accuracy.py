PLOT = False

import sys
import numpy as np
from enum import Enum
if PLOT:
    import matplotlib.pyplot as plt

from project import src
from src.settings import paths
from src.helper.pickle import load_object
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.corrector.beam_search.penalty_holder import PenaltyHolder


class CaseOperation(Enum):
    INSERTION = 0
    DELETION = 1


class Case:
    def __init__(self,
                 sequence_number: int,
                 position: int,
                 operation: CaseOperation,
                 label: bool,
                 score_difference: float):
        self.sequence_no = sequence_number
        self.position = position
        self.operation = operation
        self.label = label
        self.score_difference = score_difference

    def __lt__(self, other):
        return self.score_difference < other.score_difference


if __name__ == "__main__":
    model_name = sys.argv[1]  # "fwd1024_noise0.2"
    labeling_model_name = sys.argv[2]  # "labeling_noisy_ce"
    benchmark_name = sys.argv[3]  # "nastase-500.split"
    n = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    approach = sys.argv[5] if len(sys.argv) > 5 else ""  # "BS-fwd wikipedia"
    stepsize = 0.1

    EPSILON = 1e-16
    lookahead = 2
    labeling = labeling_model_name != "0"
    title = "%s (%s)" % (approach, benchmark_name)

    #BENCHMARKS = ["0_0.1", "0.1_0.1", "arxiv-910k", "nastase-big"]
    BENCHMARKS = [benchmark_name]

    all_insertion_intervals = []
    all_deletion_intervals = []

    for benchmark in BENCHMARKS:
        cases_path = paths.CASES_FILE_NOISY if benchmark.startswith("0.1") else paths.CASES_FILE_CLEAN
        cases_path = cases_path % (model_name, "wikipedia" if benchmark.startswith("0") else benchmark)

        sequence_cases = load_object(cases_path)

        print(len(sequence_cases))

        if labeling_model_name != "0":
            from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
            labeling_model = BidirectionalLabelingEstimator()
            labeling_model.load(labeling_model_name)

        benchmark = Benchmark(benchmark, Subset.TUNING)
        case_db = []

        correct_sequences = benchmark.get_sequences(BenchmarkFiles.CORRECT)
        corrupt_sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)

        for s_i, (correct, corrupt) in enumerate(zip(correct_sequences, corrupt_sequences)):
            if s_i == n:
                break

            print(benchmark.name, s_i)
            cases = sequence_cases[s_i]
            case_db.append([])

            labeling_space_probs = labeling_model.predict(correct.replace(' ', ''))["probabilities"] if labeling \
                else None

            correct_pos = corrupt_pos = labeling_pos = 0

            for i, case in enumerate(cases):
                true_space = correct[correct_pos] == ' ' if correct_pos < len(correct) else False
                input_space = corrupt[corrupt_pos] == ' ' if corrupt_pos < len(corrupt) else False

                space_score = np.sum(np.log([case.p_space] + [p for p in case.p_after_space[:lookahead]]))
                no_space_score = np.sum(np.log(case.p_after_no_space[:lookahead]))

                if labeling:
                    p_space_labeling = labeling_space_probs[labeling_pos]
                    p_nospace_labeling = 1 - p_space_labeling
                    p_space_labeling += EPSILON
                    p_nospace_labeling += EPSILON
                    #print(labeling_pos, p_space_labeling)
                    space_score += np.log(p_space_labeling)
                    no_space_score += np.log(p_nospace_labeling)

                op = None
                if input_space:
                    # deletion
                    op = CaseOperation.DELETION
                    score_diff = no_space_score - space_score
                    label = not true_space
                else:
                    # insertion
                    previous_space = correct_pos > 0 and correct[correct_pos - 1] == ' '
                    if not (labeling and previous_space):
                        op = CaseOperation.INSERTION
                        score_diff = space_score - no_space_score
                        label = true_space

                if op is not None:
                    eval_case = Case(s_i, corrupt_pos, op, label, score_diff)
                    case_db[-1].append(eval_case)

                correct_pos += 1
                if true_space == input_space:
                    corrupt_pos += 1
                elif input_space:
                    corrupt_pos += 2
                if not true_space:
                    labeling_pos += 1

        insertion_intervals = []
        deletion_intervals = []
        for s_i, sequence_cases in enumerate(case_db):
            print(corrupt_sequences[s_i])
            for op_type in list(CaseOperation):
                op_cases = [c for c in sequence_cases if c.operation == op_type]
                true_ops = [c for c in op_cases if c.label]
                false_ops = [c for c in op_cases if not c.label]
                min_true = np.inf if len(true_ops) == 0 else min(true_ops).score_difference
                max_false = 0 if len(false_ops) == 0 else max(false_ops).score_difference
                print(len(true_ops), len(false_ops), (max_false, min_true))
                interval = (max_false, min_true)
                if op_type == CaseOperation.INSERTION:
                    insertion_intervals.append(interval)
                else:
                    deletion_intervals.append(interval)
        weight = 20 if benchmark.name == "nastase-big" else 1
        for _ in range(weight):
            all_insertion_intervals.extend(insertion_intervals)
            all_deletion_intervals.extend(deletion_intervals)

        if PLOT:
            fig, axs = plt.subplots(2, 2)
            fig.suptitle(title)

            for op_i, op_type in enumerate(list(CaseOperation)):
                op_cases = []
                for sequence_cases in case_db:
                    for case in sequence_cases:
                        if case.operation == op_type:
                            op_cases.append(case)
                    #op_cases.extend([case for case in seq_ops if case.operation == op_type])
                n_true = len([case for case in op_cases if case.label])
                print(op_type, len(op_cases), n_true)

                n_tp = 0
                n_fp = 0

                penalties = []
                precisions = []
                recalls = []
                f_scores = []

                for c in sorted(op_cases, reverse=True):
                    if c.score_difference <= 0:
                        break
                    #print(c.score_difference, c.label)
                    if c.label:
                        n_tp += 1
                    else:
                        n_fp += 1
                    n_fn = n_true - n_tp
                    if c.score_difference < np.inf:
                        prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
                        rec = n_tp / n_true
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                        penalties.append(c.score_difference)
                        precisions.append(prec)
                        recalls.append(rec)
                        f_scores.append(f1)

                #for i in range(len(penalties)):
                #    print(penalties[i], precisions[i], recalls[i], f_scores[i])

                axs[op_i][0].plot(penalties, precisions, "--k", label="precision")
                axs[op_i][0].plot(penalties, recalls, ":k", label="recall")
                axs[op_i][0].plot(penalties, f_scores, "-k", label="F-score")

            intervals = insertion_intervals if op_type == CaseOperation.INSERTION else deletion_intervals
            n_intervals = len(intervals)
            nonempty_intervals = [(begin, end) for begin, end in intervals if end > begin]
            print(op_type, n_intervals, len(nonempty_intervals))
            sequence_cases = []
            for begin, end in nonempty_intervals:
                sequence_cases.append((end, 1))
                sequence_cases.append((begin, -1))
            sequence_penalties = []
            sequence_accuracies = []
            n_correct = 0
            for penalty, effect in sorted(sequence_cases, reverse=True):
                if penalty <= 0:
                    break
                n_correct += effect
                if penalty < np.inf:
                    sequence_penalties.append(penalty)
                    sequence_accuracies.append(n_correct / n_intervals)
            axs[op_i][1].plot(sequence_penalties, sequence_accuracies, "-r", label="seq. acc.")

            axs[0][0].legend()
            axs[0][1].legend()
            axs[0][0].set_title("insertions")
            axs[0][1].set_title("insertions")
            axs[1][0].set_title("deletions")
            axs[1][1].set_title("deletions")
            plt.show()

    n_intervals = len(all_insertion_intervals)
    print(n_intervals, "total intervals")
    nonempty_interval_pairs = [(insertion_interval, deletion_interval) for insertion_interval, deletion_interval
                               in zip(all_insertion_intervals, all_deletion_intervals)
                               if insertion_interval[0] < insertion_interval[1]
                               and deletion_interval[0] < deletion_interval[1]]
    print(len(nonempty_interval_pairs), "nonempty interval pairs")
    insertion_penalties = np.arange(0, 20, stepsize)
    deletion_penalties = np.arange(0, 20, stepsize)
    best_acc = 0
    best_penalties = (0, 0)
    heatmap = np.zeros((len(insertion_penalties), len(deletion_penalties)))
    for i, insertion_penalty in enumerate(insertion_penalties):
        for j, deletion_penalty in enumerate(deletion_penalties):
            n_correct = 0
            for insertion_interval, deletion_interval in nonempty_interval_pairs:
                if insertion_interval[0] < insertion_penalty < insertion_interval[1] \
                        and deletion_interval[0] < deletion_penalty < deletion_interval[1]:
                    n_correct += 1
            acc = n_correct / n_intervals
            heatmap[i, j] = acc
            print("%.1f" % insertion_penalty, "%.1f" % deletion_penalty, n_correct / n_intervals)
            if acc > best_acc or (acc == best_acc and
                                  insertion_penalty + deletion_penalty < best_penalties[0] + best_penalties[1]):
                best_acc = acc
                best_penalties = (insertion_penalty, deletion_penalty)

    print("best acc = %f at (%f, %f)" % (best_acc, best_penalties[0], best_penalties[1]))

    if PLOT:
        fig, ax = plt.subplots()
        fig.suptitle(title)

        image = ax.imshow(heatmap[::-1, :], extent=[0, 20, 0, 20])
        fig.colorbar(image, ax=ax)
        ax.set_xlabel("deletion penalty")
        ax.set_ylabel("insertion penalty")
        plt.show()

    holder = PenaltyHolder(seq_acc=True)
    key = model_name
    if labeling_model_name != "0":
        key += "_" + labeling_model_name
    holder.set(key, benchmark_name, -best_penalties[0], -best_penalties[1])
    print("Saved penalties.")
