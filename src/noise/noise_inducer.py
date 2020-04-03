import abc
import random


class NoiseInducer:
    def __init__(self, seed: int):
        self.rdm = random.Random(seed)

    @abc.abstractmethod
    def induce_noise(self, sequence: str) -> str:
        raise NotImplementedError()
