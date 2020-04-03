from typing import Optional

from src.models.char_lm.load import load_model
from src.corrector.threshold_holder import ThresholdHolder
from src.corrector.greedy_corrector import GreedyCorrector
from src.corrector.best_first_search_corrector import BestFirstSearchCorrector


def load_corrector(model_name: Optional[str]=None,
                   fwd_model_name: Optional[str]=None,
                   bwd_model_name: Optional[str]=None,
                   algorithm: str="greedy",
                   patience_steps: int=0,
                   noise_type: str="spaces"):
    """
    Loads a corrector wrapper.
    The used model is a bidirectional model, if model_name is specified,
    and a combined model otherwise.
    Stored threshold values are retrieved for the given noise type.
    
    :param model_name: The name of the bidirectional model.
    :param fwd_model_name: The name of the forward model.
    :param bwd_model_name: The name of the backward model.
    :param algorithm: Choose search strategy from {"greedy", "bfs"}.
    """
    # load model:
    if model_name is not None:
        model = load_model(type="bidir",
                           name=model_name)
    else:
        model = load_model(type="combined",
                           fwd=fwd_model_name,
                           bwd=bwd_model_name,
                           method="conjunction",
                           normalize=False,
                           compare=True)
    # make corrector from model:
    if algorithm == "greedy":
        # load thresholds:
        threshold_holder = ThresholdHolder()
        insertion_threshold, deletion_threshold = threshold_holder.get_thresholds(
            model_name=model_name,
            fwd_model_name=fwd_model_name,
            bwd_model_name=bwd_model_name,
            noise_type=noise_type
        )
        corrector = GreedyCorrector(model,
                                    insertion_threshold=insertion_threshold,
                                    deletion_threshold=deletion_threshold)
    elif algorithm == "bfs":
        corrector = BestFirstSearchCorrector(model, patience_steps)
    else:
        raise Exception("Algorithm '%s' not implemented." % algorithm)
    return corrector
