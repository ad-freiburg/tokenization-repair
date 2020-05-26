from typing import List


class Case:
    def __init__(self,
                 sequence_index: int,
                 position: int,
                 true_space: bool,
                 p_space: float,
                 p_after_space: List[float],
                 p_after_no_space: List[float]):
        self.sequence_index = sequence_index
        self.position = position
        self.true_space = true_space
        self.p_space = p_space
        self.p_after_space = p_after_space
        self.p_after_no_space = p_after_no_space
