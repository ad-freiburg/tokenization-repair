from typing import Dict, List, Union, Tuple

import numpy as np

from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.sequence.functions import get_space_positions_in_merged


class Beam:
    def __init__(self,
                 state: Dict,
                 log_likelihood: float):
        self.state = state
        self.log_likelihood = log_likelihood

    def sequence_length(self):
        return len(self.state["sequence"]) - 1  # do not count SOS


class BeamSearchCorrector:
    def __init__(self,
                 model: UnidirectionalLMEstimator,
                 backward: bool,
                 n_beams: int,
                 penalty: Union[float, Tuple[float, float]],
                 average_log_likelihood: bool):
        self.model = model
        self.backward = backward
        self.n_beams = n_beams
        self.average_log_likelihood = average_log_likelihood
        self.space_label = self.model.encoder.encode_char(' ')
        if isinstance(penalty, float):
            penalty = 0 if penalty == 0 else np.log(penalty)
            self.insertion_penalty = penalty
            self.deletion_penalty = penalty
        else:
            self.insertion_penalty, self.deletion_penalty = penalty

    def _empty_beam(self):
        return Beam(state=self.model.initial_state(),
                    log_likelihood=0)

    def _expand_beam(self, beam: Beam, x: int):
        prob = 1 if "probabilities" not in beam.state else beam.state["probabilities"][x]
        log_likelihood = beam.log_likelihood + np.log(prob)
        new_state = self.model.step(beam.state, x)
        return Beam(state=new_state,
                    log_likelihood=log_likelihood)

    def _score(self, beam: Beam):
        score = beam.log_likelihood
        if self.average_log_likelihood:
            score = score / beam.sequence_length()
        return score

    def _sort_beams(self, beams: List[Beam]):
        return sorted(beams, key=lambda beam: self._score(beam), reverse=True)

    def _best_beams(self, beams: List[Beam]) -> List[Beam]:
        beams = self._sort_beams(beams)
        beams = beams[:self.n_beams]
        return beams

    def _decode_predicted_sequence(self, characters: str, labels: List[int]):
        labels = labels[1:-1]
        label_index = 0
        predicted = ""
        for char in characters:
            if labels[label_index] == self.space_label:
                predicted += ' '
                label_index += 2
            else:
                label_index += 1
            predicted += char
        return predicted

    @staticmethod
    def _get_space_positions(sequence: str):
        return get_space_positions_in_merged(sequence)

    @staticmethod
    def _punish_beam(beam: Beam, penalty: float):
        beam.log_likelihood += penalty

    def correct(self, sequence: str, verbose: bool = True):
        space_positions = self._get_space_positions(sequence[::-1] if self.backward else sequence)
        sequence = sequence.replace(' ', '')
        encoded = self.model.encoder.encode_sequence(sequence)
        if self.backward:
            encoded = encoded[::-1]
            sequence = sequence[::-1]
        beams = [self._empty_beam()]
        for step, label in enumerate(encoded):
            new_beams = []
            for beam in beams:
                if step > 0:  # no space before SOS
                    space_beam = self._expand_beam(beam, self.space_label)
                    space_beam = self._expand_beam(space_beam, label)
                    if step - 1 not in space_positions:
                        self._punish_beam(space_beam, self.insertion_penalty)
                    new_beams.append(space_beam)
                no_space_beam = self._expand_beam(beam, label)
                if step > 0 and step - 1 in space_positions:
                    self._punish_beam(no_space_beam, self.deletion_penalty)
                new_beams.append(no_space_beam)
            beams = self._best_beams(new_beams)
            if verbose:
                print(self.model.encoder.decode_label(label))
                for beam in beams:
                    print("%.2f" % self._score(beam),
                          "%.2f" % beam.log_likelihood,
                          self.model.encoder.decode_sequence(beam.state["sequence"]))
        predicted = self._decode_predicted_sequence(sequence, beams[0].state["sequence"])
        if self.backward:
            predicted = predicted[::-1]
        return predicted
