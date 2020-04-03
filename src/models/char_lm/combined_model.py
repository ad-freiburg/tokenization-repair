import numpy as np

from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.load.old_models import load_old_fwd_model, load_old_bwd_model
from src.models.wrapper.unidirectional_wrapper import UnidirectionalWrapper
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.encoding.one_hot import labels
from src.encoding.character_encoder import get_encoder
from src.helper.data_structures import gather


def conjunction(p1, p2):
    return p1 * p2


def mean(p1, p2):
    return (p1 + p2) / 2


def disjunction(p1, p2):
    return p1 + (1 - p1) * p2


def normalize_probs(p):
    z = np.sum(p, axis=-1)
    normalized = (p.T / z).T
    return normalized


def combine_probabilities(p1, p2, method):
    if method == "conjunction":
        p = conjunction(p1, p2)
    elif method == "disjunction":
        p = disjunction(p1, p2)
    elif method == "mean":
        p = mean(p1, p2)
    else:
        raise NotImplementedError()
    return p


def combine_compared(p1, p2, method, labels1, labels2):
    p_compare1 = gather(p1, labels1)
    p_compare2 = gather(p2, labels2)
    p_compare = combine_probabilities(p_compare1, p_compare2, method)
    p_combined = combine_probabilities(p1, p2, method)
    p_compared = (p_combined.T / (p_combined.T + p_compare)).T
    return p_compared


class CombinedModel(CharacterLanguageModel):
    def __init__(self,
                 old_models=False,
                 forward_name=None,
                 backward_name=None,
                 method="conjunction",
                 normalize=True,
                 compare=True):
        self.old_models = old_models
        if old_models:
            fwd_model = load_old_fwd_model()
            bwd_model = load_old_bwd_model()
            self.fwd_model = UnidirectionalWrapper(fwd_model, backward=False)
            self.bwd_model = UnidirectionalWrapper(bwd_model, backward=True)
        else:
            self.fwd_model = UnidirectionalLMEstimator()
            self.fwd_model.load(forward_name)
            self.bwd_model = UnidirectionalLMEstimator()
            self.bwd_model.load(backward_name)
        self.method = method
        self.normalize = normalize
        self.compare = compare
        self.encoder = self.fwd_model.encoder if not old_models else get_encoder()

    def get_encoder(self):
        return self.encoder

    def _combine_probabilities(self, p1, p2):
        probabilities = combine_probabilities(p1, p2, self.method)
        if self.normalize:
            probabilities = normalize_probs(probabilities)
        return probabilities

    def _combine_compared(self, p1, p2, labels1, labels2):
        return combine_compared(p1, p2, self.method, labels1, labels2)

    def _compute_probabilities(self, forward_probabilities, backward_probabilities, labels):
        combined_probs = self._combine_probabilities(forward_probabilities[:-1, :], backward_probabilities[1:, :])
        if self.compare:
            insertion_probs = self._combine_compared(forward_probabilities, backward_probabilities,
                                                     labels1=labels[1:], labels2=labels[:-1])
            compared_probs = self._combine_compared(forward_probabilities[:-1, :], backward_probabilities[1:, :],
                                                    labels1=labels[2:], labels2=labels[:-2])
            sequence_probs = gather(compared_probs, labels[1:-1])
            deletion_probs = 1 - sequence_probs
        else:
            insertion_probs = self._combine_probabilities(forward_probabilities, backward_probabilities)
            deletion_probs = 1 - combined_probs
        return combined_probs, insertion_probs, deletion_probs

    def _model_predict(self, sequence):
        if self.old_models:
            fwd_probs = self.fwd_model.predict(sequence)
            bwd_probs = self.bwd_model.predict(sequence)
            sequence_labels = labels(sequence, self.fwd_model.encoder)
        else:
            fwd_prediction = self.fwd_model.predict(sequence)
            bwd_prediction = self.bwd_model.predict(sequence)
            fwd_probs = fwd_prediction["probabilities"]
            bwd_probs = bwd_prediction["probabilities"]
            sequence_labels = fwd_prediction["labels"]
        p, p_ins, p_del = self._compute_probabilities(fwd_probs, bwd_probs, sequence_labels)
        return {"probabilities": p,
                "insertion_probabilities": p_ins,
                "deletion_probabilities": p_del,
                "forward_probabilities": fwd_probs[:-1, :],
                "backward_probabilities": bwd_probs[1:, :],
                "labels": sequence_labels[1:-1]}
