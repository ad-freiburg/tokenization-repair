from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.load.old_models import load_old_fwd_model, load_old_bwd_model
from src.models.wrapper.unidirectional_wrapper import UnidirectionalWrapper
from src.settings import paths
from src.settings.settings import get_settings
from src.helper.pickle import load_object
from src.encoding.one_hot import encode, labels


class OldUnidirectionalModel(CharacterLanguageModel):
    def __init__(self, backward):
        settings = get_settings()
        model = load_old_bwd_model() if backward else load_old_fwd_model()
        self.model_wrapper = UnidirectionalWrapper(model, backward=backward)
        self.backward = backward
    
    def _model_predict(self, sequence):
        probabilities = self.model_wrapper.predict(sequence)
        if self.backward:
            probabilities = probabilities[1:, :]
        else:
            probabilities = probabilities[:-1, :]
        return_dict = {"probabilities": probabilities,
                       "labels": labels(sequence, self.model_wrapper.encoder)[1:-1]}
        return return_dict

