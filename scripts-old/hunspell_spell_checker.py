from typing import List, Iterator

import hunspell
import argparse
import re

from project import src
from src.helper.files import read_lines


def word_tokenize(sentence: str) -> List[str]:
    tokens = re.findall("\w+|\W", sentence)
    return tokens


class HunspellSpellChecker:
    def __init__(self):
        hunspell_path = "/usr/share/hunspell/"
        self.spell_checker = hunspell.HunSpell(hunspell_path + "en_US.dic",
                                               hunspell_path + "en_US.aff")

    def correct(self, sequence: str) -> str:
        tokens = word_tokenize(sequence)
        corrected = []
        for t in tokens:
            if (not t.isalpha()) or self.spell_checker.spell(t):
                corrected.append(t)
            else:
                suggestions = self.spell_checker.suggest(t)
                if len(suggestions) > 0:
                    corrected.append(suggestions[0])
                else:
                    corrected.append(t)
        sequence = "".join(corrected)
        return sequence


def interactive_sequences() -> Iterator[str]:
    while True:
        text = input("> ")
        if text == "exit":
            break
        else:
            yield text


def main(args):
    corrector = HunspellSpellChecker()
    if args.input is None:
        sequences = interactive_sequences()
    else:
        sequences = read_lines(args.input)
    for sequence in sequences:
        corrected = corrector.correct(sequence)
        print(corrected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=False)
    args = parser.parse_args()
    main(args)
