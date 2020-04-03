from src.corrector.token_corrector import TokenCorrector
from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.helper.data_structures import argmax_and_max
from src.sequence.predicted_sequence import PredictedSequence


def score(p, threshold):
    return (p - threshold) / (1 - threshold) if threshold > 0 else p


class GreedyCorrector(TokenCorrector):
    def __init__(self,
                 model: CharacterLanguageModel,
                 insertion_threshold: float,
                 deletion_threshold: float,
                 insert=True,
                 delete=True):
        super(GreedyCorrector, self).__init__(model, insert=insert, delete=delete)
        self.insertion_threshold = insertion_threshold
        self.deletion_threshold = deletion_threshold

    def correct(self, sequence, return_details=False):
        terminated = False
        predicted_sequence = PredictedSequence(sequence)
        while not terminated:
            print(sequence)
            probs = self._get_probabilities(sequence)
            if self.insert:
                insertion_probs = probs["insertion"]
                insertion_probs = predicted_sequence.prevent_double_insertion(insertion_probs)
                for i, char in enumerate(sequence):
                    if char == ' ':
                        insertion_probs[i] = 0
                        insertion_probs[i + 1] = 0
                insert_pos, p_insert = argmax_and_max(insertion_probs)
                insert_score = score(p_insert, self.insertion_threshold)
            else:
                p_insert = -1
                insert_pos = -1
                insert_score = -1
            if self.delete:
                deletion_probs = probs["deletion"]
                deletion_probs = predicted_sequence.prevent_double_deletion(deletion_probs)
                delete_pos, p_delete = argmax_and_max(deletion_probs)
                delete_score = score(p_delete, self.deletion_threshold)
            else:
                p_delete = -1
                delete_pos = -1
                delete_score = -1
            print("insert:", (insert_score, p_insert, insert_pos), "delete:", (delete_score, p_delete, delete_pos))
            if insert_score >= 0 and (delete_score <= 0 or insert_score > delete_score):
                sequence = predicted_sequence.insert(insert_pos, probability=p_insert)
            elif delete_score >= 0:
                sequence = predicted_sequence.delete(delete_pos, probability=p_delete)
            else:
                terminated = True
        if return_details:
            return predicted_sequence
        return sequence
