from typing import Dict, List, Set, Optional
import numpy as np

from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator


def space_positions_in_merged(sequence: str) -> Set[int]:
    space_positions = set()
    for i, char in enumerate(sequence):
        if char == ' ':
            space_positions.add(i - len(space_positions))
    return space_positions


class Beam:
    def __init__(self, cell_state, sequence, logprob):
        self.cell_state = cell_state
        self.sequence = sequence
        self.logprob = logprob

    def __str__(self):
        return "Beam(%s, %f)" % (self.sequence, self.logprob)

    def __repr__(self):
        return str(self)


class BatchedBeamSearchCorrector:
    def __init__(self,
                 model: UnidirectionalLMEstimator,
                 insertion_penalty: float,
                 deletion_penalty: float,
                 n_beams: int,
                 verbose: bool = False,
                 labeling_model: Optional[BidirectionalLabelingEstimator] = None):
        self.model = model
        self.backward = model.specification.backward
        self.insertion_penalty = insertion_penalty
        self.deletion_penalty = deletion_penalty
        self.n_beams = n_beams
        self.space_label = model.encoder.encode_char(' ')
        self.verbose = verbose
        self.labeling_model = labeling_model

    def _start_beam(self):
        initial_state = self.model.initial_state()["cell_state"]
        cell_state = {key: initial_state[key][0] for key in initial_state}
        beam = Beam(cell_state, "", 0)
        return beam

    def _make_input_dict(self, beams: List[Beam], x: int) -> Dict:
        n_beams = len(beams)
        input_dict = {"x": np.zeros(shape=(2 * n_beams, 2), dtype=int),
                      "sequence_lengths": [1, 2] * n_beams}
        cell_state_keys = [key for key in beams[0].cell_state]
        for key in cell_state_keys:
            dim = len(beams[0].cell_state[key])
            input_dict[key] = np.zeros(shape=(2 * n_beams, dim))
        for b_i, beam in enumerate(beams):
            input_dict["x"][2 * b_i, :] = [x, -1]
            input_dict["x"][2 * b_i + 1, :] = [x, self.space_label]
            for key in cell_state_keys:
                input_dict[key][2 * b_i, :] = beam.cell_state[key]
                input_dict[key][2 * b_i + 1, :] = beam.cell_state[key]
        if self.backward:
            input_dict["x"] = input_dict["x"][:, ::-1]
        return input_dict

    def _best_beams(self, beams: List[Beam]):
        beams = sorted(beams, key=lambda beam: beam.logprob, reverse=True)
        return beams[:self.n_beams]

    def _get_cell_state(self, result_dict, cell_state_keys, index):
        cell_state = {key: result_dict[key][index, :] for key in cell_state_keys}
        return cell_state

    def _expand_beams(self,
                      beams: List[Beam],
                      x: int,
                      y: int,
                      next_character: str,
                      original_space: bool,
                      combined_model_space_prob: Optional[float] = None) -> List[Beam]:
        cell_state_keys = [key for key in beams[0].cell_state]
        input_dict = self._make_input_dict(beams, x)
        result = self.model.predict_fn(input_dict)
        probabilities = result["probabilities"]
        if self.backward:
            probabilities = probabilities[:, ::-1, :]
        combined_log_p_space = 0 if combined_model_space_prob is None else np.log(combined_model_space_prob)
        combined_log_p_no_space = 0 if combined_model_space_prob is None else np.log(1 - combined_model_space_prob)
        new_beams = []
        for b_i, beam in enumerate(beams):
            # no space beam:
            p_no_space = probabilities[2 * b_i, 0, y]
            logprob = beam.logprob + np.log(p_no_space) + combined_log_p_no_space
            if original_space:
                logprob += self.deletion_penalty
            no_space_beam = Beam(
                cell_state=self._get_cell_state(result, cell_state_keys, 2 * b_i),
                sequence=beam.sequence + next_character,
                logprob=logprob
            )
            new_beams.append(no_space_beam)
            # space beam:
            p_space = probabilities[2 * b_i + 1, 0, self.space_label]
            p_after_space = probabilities[2 * b_i + 1, 1, y]
            logprob = beam.logprob + np.log(p_space) + np.log(p_after_space) + combined_log_p_space
            if not original_space:
                logprob += self.insertion_penalty
            space_beam = Beam(
                cell_state=self._get_cell_state(result, cell_state_keys, 2 * b_i + 1),
                sequence=beam.sequence + ' ' + next_character,
                logprob=logprob
            )
            new_beams.append(space_beam)
        return new_beams

    def correct(self, sequence: str) -> str:
        merged = sequence.replace(' ', '')
        encoded = self.model.encoder.encode_sequence(merged)
        if self.backward:
            merged = merged[::-1]
            encoded = encoded[::-1]
        original_spaces_in_merged = space_positions_in_merged(sequence[::-1] if self.backward else sequence)

        if self.labeling_model is not None:
            labeling_space_probs = self.labeling_model.predict(merged)["probabilities"]

        start_beam = self._start_beam()
        beams = [start_beam]

        for i in range(len(encoded) - 1):
            beams = self._expand_beams(beams,
                                       x=encoded[i],
                                       y=encoded[i + 1],
                                       next_character=merged[i] if i < len(merged) else '',
                                       original_space=i in original_spaces_in_merged,
                                       combined_model_space_prob=
                                       None if self.labeling_model is None else labeling_space_probs[i])
            beams = self._best_beams(beams)
            if self.verbose:
                print("step %i, symbol = %s" % (i, self.model.encoder.decode_label(encoded[i + 1])))
                print(encoded[i], encoded[i + 1])
                for beam in beams:
                    print("%.4f %s" % (beam.logprob, beam.sequence))

        predicted_sequence = beams[0].sequence
        if self.backward:
            predicted_sequence = predicted_sequence[::-1]

        return predicted_sequence
