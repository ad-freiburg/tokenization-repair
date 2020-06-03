from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.settings import model_names


def load_unidirectional_model(backward: bool,
                              robust: bool) -> UnidirectionalLMEstimator:
    name = model_names.unidirectional_model_name(backward, robust)
    model = UnidirectionalLMEstimator()
    model.load(name)
    return model


def load_bidirectional_model(robust: bool) -> BidirectionalLabelingEstimator:
    name = model_names.bidirectional_model_name(robust)
    model = BidirectionalLabelingEstimator()
    model.load(name)
    return model
