from typing import Dict, List, Tuple, Optional, Any

import random

from src.noise.noise_inducer import NoiseInducer
from src.helper.files import read_lines
from src.settings import paths


def read_error_dict(tsv_file: str) -> Dict[str, List[Tuple[str, int]]]:
    errors = {}
    for line in read_lines(tsv_file):
        wrong, correct, freq = line.split("\t")
        freq = int(freq)
        if len(correct) <= 3 and " " not in wrong and " " not in correct:
            if correct not in errors:
                errors[correct] = [(wrong, freq)]
            else:
                errors[correct].append((wrong, freq))
    return errors


def toss_coin(rdm: random.Random, p: float) -> bool:
    return rdm.random() < p


def sample(rdm: random.Random, lst: List[Tuple[Any, int]]) -> Optional[Any]:
    total = sum(freq for _, freq in lst)
    threshold = rdm.random() * total
    accumulated = 0
    for elem, freq in lst:
        accumulated += freq
        if accumulated >= threshold:
            return elem
    return None


class ACLNoiseInducer(NoiseInducer):
    def __init__(self, p: float, insertion_prob: float, seed: int):
        super().__init__(seed)
        self.error_dict = read_error_dict(paths.OCR_ERROR_FREQUENCIES_FILE)
        self.p = p
        self.insertion_prob = insertion_prob

    def toss_coin(self):
        return toss_coin(self.rdm, self.p)

    def sample_insertion(self):
        return sample(self.rdm, self.error_dict[""])

    def corrupt_token(self, token: str) -> str:
        corrupt = ""
        i = 0
        while i < len(token) + 1:
            if self.toss_coin():
                if toss_coin(self.rdm, self.insertion_prob):
                    corrupt += self.sample_insertion()
                    if i < len(token):
                        corrupt += token[i]
                    i += 1
                elif i < len(token):
                    errors = []
                    if token[i] in self.error_dict:
                        errors.extend([((1, err), freq) for err, freq in self.error_dict[token[i]]])
                    if i + 1 < len(token) and token[i:(i + 2)] in self.error_dict:
                        errors.extend([((2, err), freq) for err, freq in self.error_dict[token[i:(i + 2)]]])
                    if i + 2 < len(token) and token[i:(i + 3)] in self.error_dict:
                        errors.extend([((3, err), freq) for err, freq in self.error_dict[token[i:(i + 3)]]])
                    sampled_error = sample(self.rdm, errors)
                    if sampled_error is None:
                        corrupt += token[i]
                        i += 1
                    else:
                        length, replacement = sampled_error
                        corrupt += replacement
                        i += length
            elif i < len(token):
                corrupt += token[i]
                i += 1
            else:
                i += 1
        return corrupt

    def induce_noise(self, sequence: str) -> str:
        tokens = sequence.split()
        for i, token in enumerate(tokens):
            tokens[i] = self.corrupt_token(token)
        sequence = " ".join(tokens)
        return sequence
