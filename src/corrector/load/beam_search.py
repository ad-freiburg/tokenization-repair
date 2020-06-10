from typing import Tuple, Optional

from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
from src.corrector.beam_search.two_pass_corrector import TwoPassCorrector
from src.corrector.load.model import load_unidirectional_model, load_bidirectional_model
from src.corrector.beam_search.penalty_holder import PenaltyHolder
from src.benchmark.benchmark import get_benchmark_name


INF = float("inf")


def get_penalty_name(model, labeling_model):
    penalty_name = model.specification.name
    if labeling_model is not None:
        penalty_name += "_%s" % labeling_model.specification.name
    penalty_name += "_lookahead2"
    return penalty_name


def get_penalties(model,
                  labeling_model,
                  typos: bool,
                  token_errors: float,
                  two_pass: bool = False) -> Tuple[float, float]:
    if (not two_pass) and token_errors == INF:
        return 0, 0
    benchmark_name = get_benchmark_name(noise_level=0.1 if typos else 0,
                                        p=token_errors)
    penalty_name = get_penalty_name(model, labeling_model)
    holder = PenaltyHolder(two_pass=two_pass)
    penalties = holder.get(penalty_name, benchmark_name)
    return penalties


def initialise_beam_search_corrector(model, bidir_model, p_ins: float, p_del: float):
    return BatchedBeamSearchCorrector(model,
                                      insertion_penalty=p_ins,
                                      deletion_penalty=p_del,
                                      n_beams=5,
                                      verbose=True,
                                      labeling_model=bidir_model)


def load_beam_search_corrector(backward: bool,
                               robust: bool,
                               typos: bool,
                               p: float,
                               bidirectional: bool = False) -> BatchedBeamSearchCorrector:
    model = load_unidirectional_model(backward, robust)
    bidir_model = load_bidirectional_model(robust) if bidirectional else None
    p_ins, p_del = get_penalties(model, bidir_model, typos, p)
    corrector = initialise_beam_search_corrector(model, bidir_model, p_ins, p_del)
    return corrector


def load_two_pass_corrector(robust: bool,
                            typos: bool,
                            p: float,
                            forward_model: Optional[UnidirectionalLMEstimator] = None,
                            backward_model: Optional[UnidirectionalLMEstimator] = None) \
        -> TwoPassCorrector:
    if forward_model is None:
        forward_model = load_unidirectional_model(backward=False, robust=robust)
    if backward_model is None:
        backward_model = load_unidirectional_model(backward=True, robust=robust)
    p_fwd_ins, p_fwd_del = get_penalties(forward_model, labeling_model=None, typos=typos, token_errors=p)
    p_bwd_ins, p_bwd_del = get_penalties(backward_model, labeling_model=None, typos=typos, token_errors=p,
                                         two_pass=True)
    forward_corrector = initialise_beam_search_corrector(forward_model, None, p_fwd_ins, p_fwd_del)
    backward_corrector = initialise_beam_search_corrector(backward_model, None, p_bwd_ins, p_bwd_del)
    two_pass_corrector = TwoPassCorrector(forward_corrector, backward_corrector)
    return two_pass_corrector
