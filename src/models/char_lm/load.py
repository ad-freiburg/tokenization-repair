from src.models.char_lm.unidirectional_model import UnidirectionalModel
from src.models.char_lm.combined_model import CombinedModel
from src.models.char_lm.bidirectional_model import BidirectionalModel


def load_model(type, name=None, fwd=None, bwd=None, method=None, normalize=None, compare=None):
    if type == "unidir":
        return UnidirectionalModel(name)
    elif type == "bidir":
        return BidirectionalModel(name)
    elif type == "combined":
        return CombinedModel(fwd, bwd, method, normalize=normalize, compare=compare)
    else:
        raise Exception("Unknown language model type '%s'." % type)
