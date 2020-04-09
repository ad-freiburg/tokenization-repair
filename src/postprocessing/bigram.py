from typing import Optional

from src.ngram.bigram_holder import BigramHolder
from src.ngram.unigram_holder import UnigramHolder


class BigramPostprocessor:
    def __init__(self, unigrams: Optional[UnigramHolder] = None):
        if unigrams is None:
            self.unigrams = UnigramHolder()
        else:
            self.unigrams = unigrams
        self.bigrams = BigramHolder.load()

    def correct(self, sequence: str) -> str:
        snippets = sequence.split(' ')
        position = 0
        postprocessed_snippets = []
        while position < len(snippets):
            if position + 1 == len(snippets):
                postprocessed_snippets.append(snippets[position])
                break
            snippet = snippets[position]
            next = snippets[position + 1]
            merged = snippet + next
            merged_frequency = self.unigrams.get(merged)
            bigram = (snippet, next)
            bigram_frequency = self.bigrams.get(bigram)
            if merged_frequency > bigram_frequency:
                postprocessed_snippets.append(merged)
                position += 2
            else:
                postprocessed_snippets.append(snippet)
                position += 1
        postprocessed = ' '.join(postprocessed_snippets)
        return postprocessed
