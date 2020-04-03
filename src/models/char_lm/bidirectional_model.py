import numpy as np
from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.estimator.bidirectional_lm_estimator import BidirectionaLLMEstimator
from src.helper.data_structures import gather


class BidirectionalModel(CharacterLanguageModel):
    def __init__(self, model_name):
        self.model = BidirectionaLLMEstimator()
        self.model.load(model_name)

    def get_encoder(self):
        return self.model.encoder

    def _model_predict(self, sequence):
        result = self.model.predict(sequence)
        sequence_probs = gather(result["probabilities"], result["labels"])
        result["deletion_probabilities"] = 1 - sequence_probs
        return result
