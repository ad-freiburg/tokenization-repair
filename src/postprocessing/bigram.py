from typing import Optional, Tuple

from src.ngram.bigram_holder import BigramHolder
from src.ngram.unigram_holder import UnigramHolder


class BigramPostprocessor:
    def __init__(self,
                 unigrams: Optional[UnigramHolder] = None,
                 merge: bool = True,
                 split: bool = True):
        if unigrams is None:
            self.unigrams = UnigramHolder()
        else:
            self.unigrams = unigrams
        self.bigrams = BigramHolder.load()
        self.merge = merge
        self.split = split

    def _best_split(self, token: str) -> Tuple[Optional[str], int]:
        best_frequency = 0
        best_split = None
        for i in range(1, len(token)):
            bigram = token[:i], token[i:]
            bigram_frequency = self.bigrams.get(bigram)
            if bigram_frequency > best_frequency:
                best_frequency = bigram_frequency
                best_split = ' '.join(bigram)
        return best_split, best_frequency

    def correct(self, sequence: str) -> str:
        snippets = sequence.split(' ')
        position = 0
        postprocessed_snippets = []
        while position < len(snippets):
            if position + 1 == len(snippets):
                postprocessed_snippets.append(snippets[position])
                break
            snippet = snippets[position]
            # merge:
            did_merge = did_split = False
            if self.merge:
                next = snippets[position + 1]
                merged = snippet + next
                merged_frequency = self.unigrams.get(merged)
                bigram = (snippet, next)
                bigram_frequency = self.bigrams.get(bigram)
                if merged_frequency > bigram_frequency:
                    postprocessed_snippets.append(merged)
                    position += 2
                    did_merge = True
            if self.split and not did_merge and snippet.isalnum():
                split, split_frequency = self._best_split(snippet)
                snippet_frequency = self.unigrams.get(snippet)
                if split_frequency > snippet_frequency:
                    postprocessed_snippets.append(split)
                    position += 1
                    did_split = True
            if not did_merge and not did_split:
                postprocessed_snippets.append(snippet)
                position += 1
        postprocessed = ' '.join(postprocessed_snippets)
        return postprocessed
