from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.helper.data_structures import gather


class UnidirectionalModel(CharacterLanguageModel):
    def __init__(self, model_name: str):
        self.model = UnidirectionalLMEstimator()
        self.model.load(model_name)

    def get_encoder(self):
        return self.model.encoder

    def _model_predict(self, sequence):
        prediction = self.model.predict(sequence)
        probabilities = prediction["probabilities"]
        probabilities = probabilities[1:, :] if self.model.specification.backward else probabilities[:-1, :]
        labels = prediction["labels"][1:-1]
        return {"probabilities": probabilities,
                "labels": labels}
