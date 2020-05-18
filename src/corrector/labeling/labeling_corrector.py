from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.sequence.functions import get_space_positions_in_merged, remove_spaces
from src.helper.data_structures import izip


class LabelingCorrector:
    def __init__(self,
                 model_name: str,
                 insertion_threshold: float,
                 deletion_threshold: float):
        self.model = BidirectionalLabelingEstimator()
        self.model.load(model_name)
        self.insertion_threshold = insertion_threshold
        self.deletion_threshold = deletion_threshold

    def get_space_probabilities(self, sequence: str):
        result = self.model.predict(sequence)
        space_probabilities = result["probabilities"][1:]
        return space_probabilities

    def get_deletion_probabilities(self, space_positions, space_probabilities):
        return {pos: 1 - space_probabilities[pos] for pos in space_positions}

    def get_insertion_probabilities(self, space_positions, space_probabilities):
        return {pos: space_probabilities[pos] for pos in range(len(space_probabilities)) if pos not in space_positions}

    def correct(self, sequence: str) -> str:
        original_spaces = {pos + 1 for pos in get_space_positions_in_merged(sequence)}
        merged = remove_spaces(sequence)
        space_probabilities = self.get_space_probabilities(merged)
        deletion_probabilities = self.get_deletion_probabilities(original_spaces, space_probabilities)
        insertion_probabilities = self.get_insertion_probabilities(original_spaces, space_probabilities)
        predicted = ""
        for i, char in enumerate(merged):
            predicted += char
            if (i in deletion_probabilities and deletion_probabilities[i] < self.deletion_threshold) \
                    or (i in insertion_probabilities and insertion_probabilities[i] >= self.insertion_threshold):
                predicted += ' '
        return predicted
