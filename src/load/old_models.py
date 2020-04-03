from src.models.unidirectional import UniDirectionalModel
from src.models.bidirectional import BidirectionalModel
from src.settings.settings import get_settings


def load_old_fwd_model():
    settings = get_settings()
    model = UniDirectionalModel()
    model.load(settings["paths"]["fwd"])
    return model


def load_old_bwd_model():
    settings = get_settings()
    model = UniDirectionalModel()
    model.load(settings["paths"]["bwd"])
    return model


def load_old_softmax_model():
    settings = get_settings()
    model = BidirectionalModel()
    model.load(settings["paths"]["bidirectional"])
    return model


def load_old_sigmoidal_model():
    settings = get_settings()
    model = BidirectionalModel()
    model.load(settings["paths"]["bidirectional_sigmoid"])
    return model

