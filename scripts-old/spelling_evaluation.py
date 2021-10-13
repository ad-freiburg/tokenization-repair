from typing import List, Dict

import sys
from enum import Enum
from termcolor import colored

import project
from src.edit_distance.transposition_edit_distance import edit_operations, EditOperation, OperationType
from src.benchmark.benchmark import BenchmarkFiles, Benchmark, Subset
from src.datasets.wikipedia import Wikipedia
from src.helper.data_structures import izip
from src.evaluation.metrics import precision_recall_f1


class EvaluationCase(Enum):
    TRUE_POSITIVE = 0
    FALSE_POSITIVE = 1
    FALSE_NEGATIVE = 2


class EvaluatedOperation:
    def __init__(self, edit_operation: EditOperation, case: EvaluationCase):
        self.edit_operation = edit_operation
        self.case = case

    def __str__(self):
        return "EvaluatedOperation(%s, %s)" % (self.edit_operation, self.case)

    def __repr__(self):
        return str(self)


def eval_op2str(operation: EvaluatedOperation, original_character, next_character):
    type = operation.edit_operation.type
    char = operation.edit_operation.character
    if type == OperationType.INSERTION:
        return "[+%s]" % char
    elif type == OperationType.DELETION:
        return "[-%s]" % char
    elif type == OperationType.REPLACEMENT:
        return "[%s>%s]" % (original_character, char)
    elif type == OperationType.TRANSPOSITION:
        return "[%s<>%s]" % (original_character, next_character)
    else:
        return char


def evaluate_operations(ground_truth: List[EditOperation], predicted: List[EditOperation]) -> List[EvaluatedOperation]:
    ground_truth.append(None)
    predicted.append(None)
    gt_i = pred_i = 0
    evaluated = []
    while gt_i < len(ground_truth) - 1 or pred_i < len(predicted) - 1:
        gt_op = ground_truth[gt_i]
        pred_op = predicted[pred_i]
        if gt_op == pred_op:
            case = EvaluationCase.TRUE_POSITIVE
            evaluated.append(EvaluatedOperation(gt_op, case))
            gt_i, pred_i = gt_i + 1, pred_i + 1
        elif pred_op is None or (gt_op is not None and gt_op.position < pred_op.position):
            case = EvaluationCase.FALSE_NEGATIVE
            evaluated.append(EvaluatedOperation(gt_op, case))
            gt_i += 1
        else:
            case = EvaluationCase.FALSE_POSITIVE
            evaluated.append(EvaluatedOperation(pred_op, case))
            pred_i += 1
    return evaluated


def operations_at_positions(operations: List[EvaluatedOperation]) -> Dict[int, List[EvaluatedOperation]]:
    positions = {}
    for op in operations:
        pos = op.edit_operation.position
        if pos not in positions:
            positions[pos] = []
        positions[pos].append(op)
    for pos in positions:
        positions[pos] = [op for op in positions[pos] if op.edit_operation.type == OperationType.INSERTION] + \
                         [op for op in positions[pos] if op.edit_operation.type != OperationType.INSERTION]
    return positions


CASE_COLORS = {EvaluationCase.TRUE_POSITIVE: "green",
               EvaluationCase.FALSE_POSITIVE: "red",
               EvaluationCase.FALSE_NEGATIVE: "yellow"}


class SpellingEvaluator:
    def __init__(self):
        self.space_tp = self.space_fp = self.space_fn = self.char_tp = self.char_fp = self.char_fn = 0
        self.n_sequences = 0
        self.correct_sequences = 0

    def evaluate_sequence(self, correct, corrupt, predicted):
        ground_truth_ops = edit_operations(corrupt, correct, space_replace=False)
        predicted_ops = edit_operations(corrupt, predicted, space_replace=False)

        print(corrupt)
        print(correct)
        print(predicted)
        #print(ground_truth_ops)
        #print(predicted_ops)

        evaluated_ops = evaluate_operations(ground_truth_ops, predicted_ops)
        #print(evaluated_ops)

        edited_positions = operations_at_positions(evaluated_ops)
        print_str = ""
        next_char_consumed = False
        for pos in range(len(corrupt) + 1):
            char = corrupt[pos] if pos < len(corrupt) else ''
            next_char = corrupt[pos + 1] if pos + 1 < len(corrupt) else ''
            char_consumed = next_char_consumed
            next_char_consumed = False
            if pos in edited_positions:
                for op in edited_positions[pos]:
                    color = CASE_COLORS[op.case]
                    op_str = eval_op2str(op, char, next_char)
                    print_str += colored(op_str, color)
                    if op.edit_operation.type != OperationType.INSERTION:
                        char_consumed = True
                    if op.edit_operation.type == OperationType.TRANSPOSITION:
                        next_char_consumed = True
            if not char_consumed:
                print_str += char
        print(print_str)

        space_tp = space_fp = space_fn = char_tp = char_fp = char_fn = 0
        for op in evaluated_ops:
            char = op.edit_operation.character
            case = op.case
            if char == ' ':
                if case == EvaluationCase.TRUE_POSITIVE:
                    space_tp += 1
                elif case == EvaluationCase.FALSE_POSITIVE:
                    space_fp += 1
                else:
                    space_fn += 1
            else:
                if case == EvaluationCase.TRUE_POSITIVE:
                    char_tp += 1
                elif case == EvaluationCase.FALSE_POSITIVE:
                    char_fp += 1
                else:
                    char_fn += 1

        self.space_tp += space_tp
        self.space_fp += space_fp
        self.space_fn += space_fn
        self.char_tp += char_tp
        self.char_fp += char_fp
        self.char_fn += char_fn

        self.n_sequences += 1
        is_correct = predicted == correct
        if is_correct:
            self.correct_sequences += 1

        space_precision, space_recall, space_f1 = precision_recall_f1(self.space_tp, self.space_fp, self.space_fn)
        char_precision, char_recall, char_f1 = precision_recall_f1(self.char_tp, self.char_fp, self.char_fn)
        precision, recall, f1 = precision_recall_f1(self.space_tp + self.char_tp,
                                                    self.space_fp + self.char_fp,
                                                    self.space_fn + self.char_fn)
        sequence_accuracy = self.correct_sequences / self.n_sequences

        print("(sequence) %s" % ("CORRECT" if is_correct else "WRONG"))
        print("(sequence) (spaces) %i TP, %i FP, %i FN" % (space_tp, space_fp, space_fn))
        print("(sequence) (chars)  %i TP, %i FP, %i FN" % (char_tp, char_fp, char_fn))
        print("(sequence) (all)    %i TP, %i FP, %i FN" % (space_tp + char_tp, space_fp + char_fp, space_fn + char_fn))
        print("(total)    (spaces) %i TP, %i FP, %i FN" % (self.space_tp, self.space_fp, self.space_fn))
        print("(total)    (spaces) %.4f F-score (%.4f precision, %.4f recall)" %
              (space_f1, space_precision, space_recall))
        print("(total)    (chars)  %i TP, %i FP, %i FN" % (self.char_tp, self.char_fp, self.char_fn))
        print("(total)    (chars)  %.4f F-score (%.4f precision, %.4f recall)" %
              (char_f1, char_precision, char_recall))
        print("(total)    (all)    %i TP, %i FP, %i FN" %
              (self.space_tp + self.char_tp, self.space_fp + self.char_fp, self.space_fn + self.char_fn))
        print("(total)    (all)    %.4f F-score (%.4f precision, %.4f recall)" % (f1, precision, recall))
        print("(total)    %.4f sequence accuracy (%i / %i correct)" %
              (sequence_accuracy, self.correct_sequences, self.n_sequences))
        print()


if __name__ == "__main__":
    benchmark_name = sys.argv[1]
    file_name = sys.argv[2]
    n_sequences = int(sys.argv[3])

    benchmark = Benchmark(benchmark_name, Subset.DEVELOPMENT)
    correct_sequences = Wikipedia.development_sequences()
    corrupt_sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
    predicted_sequences = benchmark.get_predicted_sequences(file_name)
    n_sequences = (len(predicted_sequences) - 1) if n_sequences == -1 else n_sequences

    evaluator = SpellingEvaluator()

    for s_i, correct, corrupt, predicted in izip(correct_sequences, corrupt_sequences, predicted_sequences):
        if s_i == n_sequences:
            break
        evaluator.evaluate_sequence(correct, corrupt, predicted)
