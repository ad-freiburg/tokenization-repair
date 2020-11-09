from enum import Enum
import string

from src.noise.noise_inducer import NoiseInducer
from src.helper.stochastic import flip_coin


INSERTION_CHARACTERS = string.ascii_letters + string.digits + string.punctuation

OCR_CONFUSIONS = {
    "m": {"iii", "in", "ni", "rn", "rii", "ln", "nl", "tn", "lll"},
    "n": {"ii", "ll"},
    "d": {"cl", "(t"},
    "ü": {"ii", "ll"},
    "ff": {"tl", "tf"},
    "b": {"t)"},
    "p": {"l)", "i)", "1)"},
    "O": {"()"},
    "c": {"(:", "(-"},
    "ai": {"M"},
    "al": {"M"},
    "fo": {"tb"},
    "t": {"l,"}
}


class CorruptionType(Enum):
    INSERTION = 0
    DELETION = 1
    REPLACEMENT = 2


class OCRNoiseInducer(NoiseInducer):
    def __init__(self, seed: int, p: float):
        super().__init__(seed)
        self.p = p
        self.corruption_types = list(CorruptionType)
        self.max_confusion_length = max(len(pattern) for pattern in OCR_CONFUSIONS)
        self.unigram_replacements = [(1, char) for char in INSERTION_CHARACTERS]
        print("initialized OCRNoiseInducer with seed %i and p=%f" % (seed, p))

    def random_character(self) -> str:
        return self.rdm.choice(INSERTION_CHARACTERS)

    def random_replacement(self, chars: str) -> str:
        replacements = self.unigram_replacements
        for l in range(self.max_confusion_length):
            pattern = chars[:(l + 1)]
            if pattern in OCR_CONFUSIONS:
                pattern_len = len(pattern)
                replacements = replacements + [(pattern_len, replacement) for replacement in OCR_CONFUSIONS[pattern]]
        return self.rdm.choice(replacements)

    def induce_noise(self, sequence: str) -> str:
        noisy = ""
        pos = 0
        while pos < len(sequence):
            char = sequence[pos]
            append_char = True
            if flip_coin(self.rdm, self.p):
                corruption = self.rdm.choice(self.corruption_types)
                if corruption == CorruptionType.DELETION and char != " ":
                    append_char = False
                elif corruption == CorruptionType.INSERTION:
                    noisy += self.random_character()
                elif corruption == CorruptionType.REPLACEMENT and char != " ":
                    pattern_len, replacement = self.random_replacement(sequence[pos:(pos + self.max_confusion_length)])
                    noisy += replacement
                    append_char = False
                    pos += pattern_len - 1
            if append_char:
                noisy += char
            pos += 1
        return noisy
