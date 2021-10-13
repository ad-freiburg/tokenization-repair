import sys

import project

from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.interactive.sequence_generator import interactive_sequence_generator


if __name__ == "__main__":
    model = BidirectionalLabelingEstimator()
    model.load(sys.argv[1])
    threshold = float(sys.argv[2])
    for sequence in interactive_sequence_generator():
        sequence = sequence.replace(' ', '')
        result = model.predict(sequence)
        probabilities = result["probabilities"][1:]
        predicted = ""
        for char, p in zip(sequence, probabilities):
            print(char, p)
            predicted += char
            if p > threshold:
                predicted += ' '
        print(predicted)
