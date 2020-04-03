from typing import Optional
from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.models.char_lm.unidirectional_model import UnidirectionalModel
from src.models.char_lm.bidirectional_model import BidirectionalModel
from src.models.char_lm.combined_model import CombinedModel
from src.models.char_lm.old_unidirectional_model import OldUnidirectionalModel
from src.models.char_lm.old_bidirectional_model import OldBidirectionalModel


def load_char_lm(model_type: str,
                 model_name: Optional[str] = None,
                 bwd_model_name: Optional[str] = None):
    if model_type == "bidir":
        model = BidirectionalModel(model_name)
    elif model_type == "fwd" or model_type == "bwd":
        model = UnidirectionalModel(model_name)
    elif model_type == "combined":
        model = CombinedModel(old_models=False,
                              forward_name=model_name,
                              backward_name=bwd_model_name,
                              method="conjunction",
                              normalize=True,
                              compare=True)
    elif model_type == "fwd_old":
        model = OldUnidirectionalModel(backward=False)
    elif model_type == "bwd_old":
        model = OldUnidirectionalModel(backward=True)
    elif model_type == "combined_old":
        model = CombinedModel(old_models=True)
    elif model_type == "softmax_old":
        model = OldBidirectionalModel(sigmoidal=False)
    elif model_type == "sigmoid_old":
        model = OldBidirectionalModel(sigmoidal=True)
    else:
        raise NotImplementedError()
    return model


def load_default_char_lm(approach: str) -> CharacterLanguageModel:
    if approach == "combined":
        return load_char_lm("combined", "fwd1024", "bwd1024")
    elif approach == "softmax":
        return load_char_lm("bidir", "softmax1024")
    elif approach == "sigmoid":
        return load_char_lm("bidir", "sigmoid1024")
    elif approach == "combined_robust":
        return load_char_lm("combined", "fwd1024_noise0.2", "bwd1024_noise0.2")
    elif approach == "softmax_robust":
        return load_char_lm("bidir", "softmax1024_noise0.2")
    elif approach == "sigmoid_robust":
        return load_char_lm("bidir", "sigmoid1024_noise0.2")
    else:
        raise NotImplementedError("Unknown approach '%s'." % str(approach))