
from src.corrector.sentence_corrector import BidirectionalCorrector
from src.data.simplewiki import Simplewiki
from src.settings import paths

from bidirectional_bs import BidirectionalModel


def get_default_bidirectional_corrector():
    model = BidirectionalModel()
    model.load(paths.BIDIRECTIONAL_MODEL_DIR)
    encoder, decoder = Simplewiki.get_dictionaries()
    corrector = BidirectionalCorrector(model, encoder, decoder_dict=decoder)
    return corrector
