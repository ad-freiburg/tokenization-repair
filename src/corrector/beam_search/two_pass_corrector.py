from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector


class TwoPassCorrector:
    def __init__(self,
                 forward_corrector: BatchedBeamSearchCorrector,
                 backward_corrector: BatchedBeamSearchCorrector):
        self.forward_corrector = forward_corrector
        self.backward_corrector = backward_corrector

    def correct(self, sequence):
        sequence = self.forward_corrector.correct(sequence)
        sequence = self.backward_corrector.correct(sequence)
        return sequence
