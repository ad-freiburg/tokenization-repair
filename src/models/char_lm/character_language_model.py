import abc
import numpy as np
from src.encoding.character_encoder import CharacterEncoder


class CharacterLanguageModel:
    @abc.abstractmethod
    def _model_predict(self, sequence):
        """
        Call the model to estimate character probabilities and insertion probabilities for the given sequence.

        :param sequence: string of length L
        :return: dictionary with the following entries
            - probabilities: array of shape (L, D)
            - insertion_probabilites: array of shape (L + 1, D)
        """
        raise NotImplementedError()

    def predict(self, sequence):
        result = self._model_predict(sequence)
        predictions = np.argmax(result["probabilities"], axis=-1)
        result["predictions"] = predictions
        return result

    @abc.abstractmethod
    def get_encoder(self) -> CharacterEncoder:
        raise NotImplementedError()
