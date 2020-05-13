from typing import Dict, List, Optional

import project
from src.interactive.parameters import Parameter, ParameterGetter

params = [Parameter("benchmark", "-benchmark", "str"),
          Parameter("n_sequences", "-n", "int"),
          Parameter("n_beams", "-b", "int"),
          Parameter("space_penalty", "-sp", "float"),
          Parameter("char_penalty", "-cp", "float"),
          Parameter("out_file", "-f", "str")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import sys
import numpy as np

from src.interactive.sequence_generator import interactive_sequence_generator
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.helper.data_structures import top_k_indices
from src.benchmark.benchmark import BenchmarkFiles, Subset, Benchmark
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp
from src.settings import symbols


def ln(x):
    return np.log(x)


class SpellingBeam:
    def __init__(self,
                 state: Dict,
                 log_likelihood: float,
                 score: float,
                 sequence: str,
                 label: Optional[int],
                 wait_steps: int = 0):
        self.state = state
        self.log_likelihood = log_likelihood
        self.sequence = sequence
        self.score = score
        self.label = label
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
                 verbose: bool = True):
        self.model = model
        self.n_beams = n_beams
        self.branching_factor = branching_factor
        self.consecutive_insertions = consecutive_insertions
        self.char_penalty = char_penalty
        self.space_penalty = space_penalty
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

    def _extend_beam(self,
                     beam: SpellingBeam,
                     log_p: float,
                     character: str,
                     label: int,
                     space_edit: bool = False,
                     char_edit: bool = False,
                     wait_steps: int = 0) -> SpellingBeam:
        log_likelihood = beam.log_likelihood + log_p
        score = beam.score + log_p
        if space_edit:
            score -= self.space_penalty
        elif char_edit:
            score -= self.char_penalty
        sequence = beam.sequence + (character if character is not None else '')
        return SpellingBeam(beam.state, log_likelihood, score, sequence, label, wait_steps)

    def _get_best_beams(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        if len(beams) <= self.n_beams:
            return beams
        scores = [beam.score for beam in beams]
        indices = top_k_indices(scores, self.n_beams)
        best = [beams[i] for i in indices]
        return best

    def _filter_duplicates(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        sequences = set()
        filtered_beams = []
        for beam in sorted(beams):
            if beam.sequence not in sequences:
                filtered_beams.append(beam)
                sequences.add(beam.sequence)
        return filtered_beams

    def _select_beams(self, beams: List[SpellingBeam]) -> List[SpellingBeam]:
        beams = self._get_best_beams(beams)
        beams = self._filter_duplicates(beams)
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
            log_p = log_probs[label]
            c = self.model.encoder.decode_label(label)
            space_edit = label == self.space_label
            char_edit = not space_edit
            new_beam = self._extend_beam(beam, log_p, c, label, space_edit=space_edit, char_edit=char_edit)
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

    def _replacement_beams(self, beam: SpellingBeam):
        new_beams = []
        probs = self._get_probabilities(beam)
        log_probs = ln(probs)
        for label in top_k_indices(probs, self.branching_factor):
            if label != self.space_label and label not in self.non_insertable_labels:
                log_p = log_probs[label]
                c = self.model.encoder.decode_label(label)
                new_beam = self._extend_beam(beam, log_p, c, label, char_edit=True)
                new_beams.append(new_beam)
        return new_beams

    def _deletion_beam(self, beam: SpellingBeam, label) -> SpellingBeam:
        is_space = label == self.space_label
        new_beam = SpellingBeam(beam.state,
                                beam.log_likelihood,
                                beam.score - (self.space_penalty if is_space else self.char_penalty),
                                beam.sequence,
                                label=None)
        new_beam.needs_update = False
        return new_beam

    def _transposition_beam(self, beam: SpellingBeam, c: str, x: int, next_c: str, next_x: int) -> SpellingBeam:
        p = beam.state["probabilities"][next_x]
        state = model.step(beam.state, next_x, include_sequence=False)
        p_after = state["probabilities"][x]
        log_p = np.sum(np.log([p, p_after]))
        log_likelihood = float(beam.log_likelihood + log_p)
        score = float(beam.score + log_p - self.char_penalty)
        sequence = beam.sequence + next_c + c
        return SpellingBeam(state,
                            log_likelihood,
                            score,
                            sequence,
                            x,
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
                new_beams.extend(self._replacement_beams(beam))
                new_beams.append(self._deletion_beam(beam, x))
                if next_c is not None:
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
                print(beam.score, beam.log_likelihood, beam.wait_steps, beam.sequence)

    def correct(self, sequence: str) -> str:
        self.total_model_time = 0
        encoded = model.encoder.encode_sequence(sequence)
        state = model.initial_state()
        initial_beam = SpellingBeam(state, 0, 0, "", encoded[0])
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
        return beams[0].sequence


if __name__ == "__main__":
    model = UnidirectionalLMEstimator()
    model.load("fwd1024")
    corrector = SpellingBeamSearchCorrector(model,
                                            n_beams=parameters["n_beams"],
                                            branching_factor=parameters["n_beams"],
                                            consecutive_insertions=2,
                                            char_penalty=parameters["char_penalty"],
                                            space_penalty=parameters["space_penalty"])

    benchmark_name = parameters["benchmark"]
    if benchmark_name != "0":
        benchmark = Benchmark(benchmark_name, Subset.DEVELOPMENT)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        n_sequences = parameters["n_sequences"]
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + parameters["out_file"])
        corrector.verbose = False
    else:
        sequences = interactive_sequence_generator()
        n_sequences = -1
        file_writer = None

    for s_i, sequence in enumerate(sequences):
        if s_i == n_sequences:
            break
        print("sequence %i" % s_i)
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        if file_writer is not None:
            file_writer.add(predicted, runtime)
            file_writer.save()
        else:
            print(runtime)
            print(corrector.total_model_time)