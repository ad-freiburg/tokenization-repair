from typing import Tuple

from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
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


def get_penalties(model, labeling_model, typos: bool, token_errors: float) -> Tuple[float, float]:
    if token_errors == INF:
        return 0, 0
    benchmark_name = get_benchmark_name(noise_level=0.1 if typos else 0,
                                        p=token_errors)
    penalty_name = get_penalty_name(model, labeling_model)
    holder = PenaltyHolder()
    penalties = holder.get(penalty_name, benchmark_name)
    return penalties


def load_beam_search_corrector(backward: bool,
                               robust: bool,
                               typos: bool,
                               p: float,
                               bidirectional: bool = False) -> BatchedBeamSearchCorrector:
    model = load_unidirectional_model(backward, robust)
    bidir_model = load_bidirectional_model(robust) if bidirectional else None
    p_ins, p_del = get_penalties(model, bidir_model, typos, p)
    corrector = BatchedBeamSearchCorrector(model,
                                           insertion_penalty=p_ins,
                                           deletion_penalty=p_del,
                                           n_beams=5,
                                           verbose=True,
                                           labeling_model=bidir_model)
    return corrector
