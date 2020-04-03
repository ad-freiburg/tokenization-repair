from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.load.old_models import load_old_softmax_model, load_old_sigmoidal_model
from src.settings import paths
from src.helper.pickle import load_object
from src.encoding.one_hot import encode, labels
from src.encoding.character_encoder import get_encoder
from src.helper.data_structures import gather


class OldBidirectionalModel(CharacterLanguageModel):
    def __init__(self, sigmoidal: bool):
        self.model = load_old_sigmoidal_model() if sigmoidal else load_old_softmax_model()
        self.encoder = load_object(paths.WIKI_ENCODER_DICT)
        self.char_encoder = get_encoder()
    
    def _model_predict(self, sequence):
        x = encode(sequence, self.encoder)
        sequence_labels = labels(sequence, self.encoder)[1:-1]
        probabilities = self.model.predict(x)
        sequence_probs = gather(probabilities, sequence_labels)
        deletion_probabilities = 1 - sequence_probs
        insertion_probabilities = self.model.predict_insertion(x)
        return_dict = {"probabilities": probabilities,
                       "insertion_probabilities": insertion_probabilities,
                       "deletion_probabilities": deletion_probabilities,
                       "labels": sequence_labels}
        return return_dict

    def get_encoder(self):
        return self.char_encoder
