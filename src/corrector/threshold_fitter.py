from enum import Enum
import numpy as np

from src.sequence.corruption import invert_corruption
from src.sequence.predicted_sequence import PredictedSequence
from src.evaluation.sequence_evaluation import SequenceEvaluation
from src.evaluation.metrics import precision_recall_f1
from src.plot.threshold_fitting import plot_precision_recall_f1


class Label(Enum):
    TRUE_POSITIVE = 1
    FALSE_POSITIVE = 2
    FALSE_NEGATIVE = 3
    REPARATION = 4


def _label_sequence_evaluation(sequence_evaluation: SequenceEvaluation):
    probability_label_pairs = []
    ground_truth = sequence_evaluation.get_ground_truth()
    false_positives = set()
    probability_prediction_pairs = sequence_evaluation.get_ordered_prob_prediction_pairs()
    for probability, prediction in probability_prediction_pairs:
        if prediction in ground_truth:
            label = Label.TRUE_POSITIVE
            ground_truth.remove(prediction)
        else:
            inverse = invert_corruption(prediction)
            if inverse in false_positives:
                label = Label.REPARATION
                false_positives.remove(inverse)
            else:
                label = Label.FALSE_POSITIVE
                false_positives.add(prediction)
        probability_label_pairs.append((probability, label))
    for _ in ground_truth:
        probability_label_pairs.append((-1, Label.FALSE_NEGATIVE))
    return probability_label_pairs


class ThresholdFitter:
    def __init__(self):
        self.runtime = 0
        self.n_sequences = 0
        self.probability_label_pairs = []
        self.thresholds = None
        self.true_positives = None
        self.false_positives = None
        self.false_negatives = None
        self.precision = None
        self.recall = None
        self.f1 = None

    def add_example(self, correct: str, corrupt: str, predicted: PredictedSequence):
        evaluation = SequenceEvaluation(correct=correct, corrupt=corrupt, predicted_sequence=predicted)
        probability_label_pairs = _label_sequence_evaluation(evaluation)
        self.probability_label_pairs += probability_label_pairs
        self.n_sequences += 1
        self.runtime += predicted.runtime

    def _count_positives(self):
        return len([1 for _, label in self.probability_label_pairs
                    if label == Label.TRUE_POSITIVE or label == Label.FALSE_NEGATIVE])

    def fit(self):
        self.probability_label_pairs = sorted(self.probability_label_pairs, reverse=True, key=lambda x: x[0])
        probabilities = [pair[0] for pair in self.probability_label_pairs]
        labels = [pair[1] for pair in self.probability_label_pairs]
        n_positives = self._count_positives()
        self.thresholds = []
        self.true_positives = []
        self.false_positives = []
        self.false_negatives = []
        true_positives = 0
        false_positives = 0
        false_negatives = n_positives
        threshold = 1
        i = 0
        while threshold >= 0:
            while probabilities[i] >= threshold:
                label = labels[i]
                if label == Label.TRUE_POSITIVE:
                    true_positives += 1
                    false_negatives -= 1
                elif label == Label.FALSE_POSITIVE:
                    false_positives += 1
                elif label == Label.REPARATION:
                    false_positives -= 1
                else:
                    raise Exception("Something weird happened: false negatives should not have probability >= 0.")
                i += 1
                if i == len(self.probability_label_pairs):
                    break
            self.thresholds.append(threshold)
            self.true_positives.append(true_positives)
            self.false_positives.append(false_positives)
            self.false_negatives.append(false_negatives)
            if i == len(self.probability_label_pairs):
                break
            threshold = probabilities[i]
        self.precision = []
        self.recall = []
        self.f1 = []
        for t, tp, fp, fn in zip(self.thresholds, self.true_positives, self.false_positives, self.false_negatives):
            precision, recall, f1 = precision_recall_f1(tp, fp, fn)
            self.precision.append(precision)
            self.recall.append(recall)
            self.f1.append(f1)

    def plot(self, title, save=False):
        plot_precision_recall_f1(self.thresholds, self.precision, self.recall, self.f1, title=title, save=save)

    def print_best(self):
        best = np.argmax(self.f1)
        threshold = self.thresholds[best]
        f1 = self.f1[best]
        precision = self.precision[best]
        recall = self.recall[best]
        tp = self.true_positives[best]
        fp = self.false_positives[best]
        fn = self.false_negatives[best]
        print("best f1 = %.4f @ %.8f" % (f1, threshold))
        print("precision = %.4f (%i/%i)" % (precision, tp, tp + fp))
        print("recall = %.4f (%i/%i)" % (recall, tp, tp + fn))

    def get_threshold(self):
        best = np.argmax(self.f1)
        threshold = self.thresholds[best]
        return threshold
