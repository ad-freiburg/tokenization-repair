from typing import Dict, List, Optional, Tuple

import numpy as np

from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.helper.data_structures import top_k_indices
from src.helper.time import time_diff, timestamp
from src.settings import symbols


def ln(x):
    return np.log(x)


MAX_DIFF_IN_CHAR_PENALTY = 2


class SpellingBeam:
    def __init__(self,
                 state: Dict,
                 log_likelihood: float,
                 score: float,
                 sequence: str,
                 segmentation_sequence: str,
                 label: Optional[int],
                 edits_in_last_word: int,
                 wait_steps: int = 0):
        self.state = state
        self.log_likelihood = log_likelihood
        self.sequence = sequence
        if segmentation_sequence.endswith('  '):
            segmentation_sequence = segmentation_sequence[:-1]
        self.segmentation_sequence = segmentation_sequence
        self.score = score
        self.label = label
        self.edits_in_last_word = edits_in_last_word
        self.wait_steps = wait_steps
        self.needs_update = True

    def __lt__(self, other):
        if self.score != other.score:
            return self.score > other.score
        return self.wait_steps > other.wait_steps


class SpellingBeamSearchCorrector:
    def __init__(self,
                 model: UnidirectionalLMEstimator,
                 n_beams: int,
                 branching_factor: int,
                 consecutive_insertions: int,
                 char_penalty: float,
                 space_penalty: float,
                 max_edits_per_word: int = 1,
                 verbose: bool = True):
        self.model = model
        self.n_beams = n_beams
        self.branching_factor = branching_factor
        self.consecutive_insertions = consecutive_insertions
        self.char_penalty = char_penalty
        self.space_penalty = space_penalty
        self.max_edits_per_word = max_edits_per_word
        self.space_label = self.model.encoder.encode_char(' ')
        self.total_model_time = 0
        self.verbose = verbose
        self.non_insertable_labels = {self.model.encoder.encode_char(label) for label in
                                      (symbols.SOS, symbols.EOS, symbols.UNKNOWN)}

    def _get_probabilities(self, beam: SpellingBeam):
        if beam.needs_update:
            start_time = timestamp()
            beam.state = self.model.step(beam.state, beam.label, include_sequence=False)
            self.total_model_time += time_diff(start_time)
            beam.needs_update = False
        probabilities = beam.state["probabilities"]
        return probabilities

    def _editable(self, beam: SpellingBeam):
        return beam.edits_in_last_word < self.max_edits_per_word

    def _extend_beam(self,
                     beam: SpellingBeam,
                     log_p: float,
                     character: str,
                     label: int,
                     space_edit: bool = False,
                     char_edit: bool = False,
                     wait_steps: int = 0,
                     original_char: Optional[str] = None) -> SpellingBeam:
        log_likelihood = beam.log_likelihood + log_p
        score = beam.score + log_p
        if space_edit:
            score -= self.space_penalty
        elif char_edit:
            score -= self.char_penalty
        original_char = character if original_char is None else original_char
        sequence = beam.sequence + (character if character is not None else '')
        if space_edit:
            segmentation = beam.segmentation_sequence + character
        else:
            segmentation = beam.segmentation_sequence + (original_char if character is not None else '')
        if character == ' ':
            n_edits = 0
        else:
            n_edits = beam.edits_in_last_word + (1 if char_edit else 0)
        return SpellingBeam(beam.state, log_likelihood, score, sequence, segmentation, label,
                            edits_in_last_word=n_edits, wait_steps=wait_steps)

    def _get_best_beams(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        if len(beams) <= self.n_beams:
            return beams
        scores = [beam.score for beam in beams]
        indices = top_k_indices(scores, self.n_beams)
        best = [beams[i] for i in indices]
        return best

    def _filter_by_n_edits(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        return [beam for beam in beams if beam.edits_in_last_word <= self.max_edits_per_word]

    def _filter_duplicates(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        sequences = set()
        filtered_beams = []
        for beam in sorted(beams):
            if (beam.sequence, beam.wait_steps) not in sequences:
                filtered_beams.append(beam)
                sequences.add((beam.sequence, beam.wait_steps))
        return filtered_beams

    def _filter_by_score(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        best_score = beams[0].score
        threshold = best_score - MAX_DIFF_IN_CHAR_PENALTY * self.char_penalty
        return [beam for beam in beams if beam.score >= threshold]

    def _select_beams(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        beams = self._get_best_beams(beams)
        beams = self._filter_duplicates(beams)
        beams = self._filter_by_score(beams)
        return beams

    def _update_beams(self, beams: List[SpellingBeam]):
        start_time = timestamp()
        update_indices = [i for i in range(len(beams)) if beams[i].needs_update]
        if len(update_indices) > 0:
            states = [beams[i].state for i in update_indices]
            labels = [beams[i].label for i in update_indices]
            states = self.model.step_batch(states, labels)
            for i, index in enumerate(update_indices):
                beams[index].state = states[i]
                beams[index].needs_update = False
        self.total_model_time += time_diff(start_time)

    def _insertion_beams(self, beam: SpellingBeam) -> List[SpellingBeam]:
        new_beams = []
        probs = self._get_probabilities(beam)
        log_probs = ln(probs)
        for label in top_k_indices(probs, self.branching_factor):
            if label in self.non_insertable_labels:
                continue
            c = self.model.encoder.decode_label(label)
            if c.isalpha() or c == ' ':
                log_p = log_probs[label]
                space_edit = label == self.space_label
                char_edit = not space_edit
                if space_edit or self._editable(beam):
                    new_beam = self._extend_beam(beam, log_p, c, label, space_edit=space_edit, char_edit=char_edit,
                                                 original_char='')
                    new_beams.append(new_beam)
        return new_beams

    def _insertion_step(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        self._update_beams(beams)
        new_beams = []
        for beam in beams:
            if beam.wait_steps > 0:
                new_beams.append(beam)
            else:
                insertion_beams = self._insertion_beams(beam)
                new_beams.extend(insertion_beams)
        beams = beams + new_beams
        beams = self._select_beams(beams)
        return beams

    def _no_edit_beam(self, beam: SpellingBeam, c: str, x: int):
        probs = self._get_probabilities(beam)
        log_probs = ln(probs)
        log_p = log_probs[x]
        new_beam = self._extend_beam(beam, log_p, c, x)
        return new_beam

    def _replacement_beams(self, beam: SpellingBeam, char: str):
        new_beams = []
        probs = self._get_probabilities(beam)
        log_probs = ln(probs)
        for label in top_k_indices(probs, self.branching_factor):
            if label != self.space_label and label not in self.non_insertable_labels:
                c = self.model.encoder.decode_label(label)
                if c.isalpha():
                    log_p = log_probs[label]
                    new_beam = self._extend_beam(beam, log_p, c, label, char_edit=True, original_char=char)
                    new_beams.append(new_beam)
        return new_beams

    def _deletion_beam(self, beam: SpellingBeam, label, char: str) -> SpellingBeam:
        is_space = label == self.space_label
        new_beam = SpellingBeam(beam.state,
                                beam.log_likelihood,
                                beam.score - (self.space_penalty if is_space else self.char_penalty),
                                beam.sequence,
                                beam.segmentation_sequence + (char if char != ' ' else ''),
                                label=None,
                                edits_in_last_word=beam.edits_in_last_word + (0 if is_space else 1))
        new_beam.needs_update = False
        return new_beam

    def _transposition_beam(self, beam: SpellingBeam, c: str, x: int, next_c: str, next_x: int) -> SpellingBeam:
        p = beam.state["probabilities"][next_x]
        state = self.model.step(beam.state, next_x, include_sequence=False)
        p_after = state["probabilities"][x]
        log_p = np.sum(np.log([p, p_after]))
        log_likelihood = float(beam.log_likelihood + log_p)
        score = float(beam.score + log_p - self.char_penalty)
        sequence = beam.sequence + next_c + c
        segmentation = beam.segmentation_sequence + c + next_c
        return SpellingBeam(state,
                            log_likelihood,
                            score,
                            sequence,
                            segmentation,
                            x,
                            edits_in_last_word=beam.edits_in_last_word + 1,
                            wait_steps=2)

    def _character_step(self,
                        beams: List[SpellingBeam],
                        c: str,
                        x: int,
                        next_c: Optional[str],
                        next_x: Optional[int]) -> List[SpellingBeam]:
        self._update_beams(beams)
        new_beams = []
        for beam in beams:
            if beam.wait_steps > 0:
                new_beams.append(beam)
            else:
                new_beams.append(self._no_edit_beam(beam, c, x))
                char_editable = self._editable(beam) and c.isalpha() and c.islower()
                if c == ' ' or char_editable:
                    new_beams.append(self._deletion_beam(beam, x, c))
                if char_editable:
                    new_beams.extend(self._replacement_beams(beam, c))
                    if next_c is not None and next_c.isalpha():
                        new_beams.append(self._transposition_beam(beam, c, x, next_c, next_x))
        beams = self._select_beams(new_beams)
        return beams

    def _reduce_wait_steps(self, beams: List[SpellingBeam]):
        for beam in beams:
            beam.wait_steps -= 1

    def _eos_step(self, beams: List[SpellingBeam], label: int) -> List[SpellingBeam]:
        self._update_beams(beams)
        for beam in beams:
            probs = self._get_probabilities(beam)
            p = probs[label]
            log_p = ln(p)
            beam.log_likelihood += log_p
            beam.score += log_p
        return sorted(beams)

    def _print_beams(self, character: str, beams: List[SpellingBeam]):
        if self.verbose:
            print(character)
            for beam in beams[::-1]:
                print(beam.score, beam.log_likelihood, beam.wait_steps, beam.edits_in_last_word, beam.sequence)

    def correct(self, sequence: str) -> Tuple[str, str]:
        self.total_model_time = 0
        encoded = self.model.encoder.encode_sequence(sequence)
        state = self.model.initial_state()
        initial_beam = SpellingBeam(state, 0, 0, "", "", encoded[0], edits_in_last_word=0)
        beams = [initial_beam]
        for i in range(len(sequence)):
            for _ in range(self.consecutive_insertions):
                beams = self._insertion_step(beams)
            c = sequence[i]
            x = encoded[i + 1]
            next_c = sequence[i + 1] if i + 1 < len(sequence) else None
            next_x = encoded[i + 2] if i + 2 < len(encoded) else None
            beams = self._character_step(beams, c, x, next_c, next_x)
            self._reduce_wait_steps(beams)
            self._print_beams(c, beams)
        beams = self._eos_step(beams, encoded[-1])
        self._print_beams("EOS", beams)
        return beams[0].sequence, beams[0].segmentation_sequence