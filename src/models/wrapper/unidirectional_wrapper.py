from src.encoding.one_hot import encode
from src.settings import paths
from src.helper.pickle import load_object


class UnidirectionalWrapper:
    def __init__(self,
                 model,
                 backward):
        self.model = model
        self.encoder = load_object(paths.WIKI_ENCODER_DICT)
        self.backward = backward
    
    def predict(self, sequence):
        x = encode(sequence, self.encoder)
        if self.backward:
            x = x[::-1, :]
        x = x[:-1, :]
        probabilities = self.model.predict(x)
        if self.backward:
            probabilities = probabilities[::-1, :]
        return probabilities

