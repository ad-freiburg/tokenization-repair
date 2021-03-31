import textrazor
import sys
import math

from project import src
from src.helper.files import read_lines


class TextRazorSpellChecker:
    def __init__(self):
        self.client = textrazor.TextRazor(extractors=["spelling"])

    def correct(self, text):
        response = self.client.analyze(text)
        corrected = ""
        pos = 0
        for w in response.words():
            start, end = w.input_start_offset, w.input_end_offset
            corrected += text[pos:start]
            if len(w.spelling_suggestions) == 0:
                corrected += text[start:end]
            else:
                corrected += w.spelling_suggestions[0]["suggestion"]
            pos = end
        corrected += text[pos:]
        return corrected


if __name__ == "__main__":
    in_file = sys.argv[1]
    batch_size = 20

    textrazor.api_key = "aa9af0833fc32984993743c270d987e5d22b3bc3464d7d30a076d7b9"

    corrector = TextRazorSpellChecker()

    lines = read_lines(in_file)

    for batch in range(math.ceil(len(lines) / batch_size)):
        start = batch * batch_size
        end = start + batch_size
        text = "\n".join(lines[start:end])
        correction = corrector.correct(text)
        print(correction)
