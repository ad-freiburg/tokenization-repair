from typing import List

from nltk.tokenize import sent_tokenize


class NLTKSentenceSplitter:
    @staticmethod
    def split(text: str) -> List[str]:
        return sent_tokenize(text)
