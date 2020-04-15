from src.ngram.unigram_holder import UnigramHolder
from src.ngram.bigram_holder import BigramHolder
from src.ngram.tokenizer import Tokenizer
from src.postprocessing.rule_based import RuleBasedPostprocessor


class LeftToRightCorrector:
    def __init__(self):
        self.unigrams = UnigramHolder()
        self.bigrams = BigramHolder.load()
        self.tokenizer = Tokenizer()
        self.postprocessor = RuleBasedPostprocessor()

    def try_merge(self, token: str, next: str) -> bool:
        return self.unigrams.get(token + next) > self.bigrams.get((token, next))

    def best_split(self, token: str) -> str:
        best = token
        best_freqency = self.unigrams.get(token)
        best_unigram_frequency = best_freqency
        for i in range(1, len(token)):
            left, right = token[:i], token[i:]
            frequency = self.bigrams.get((left, right))
            unigram_frequency = min(self.unigrams.get(left), self.unigrams.get(right))
            if frequency > best_freqency or (frequency == best_freqency and unigram_frequency > best_unigram_frequency):
                best = left + ' ' + right
                best_freqency = frequency
                best_unigram_frequency = unigram_frequency
        return best

    def correct(self, sequence: str) -> str:
        tokens = self.tokenizer.tokenize(sequence)
        texts = [token.text for token in tokens]
        predicted = ""
        t_i = 0
        while t_i < len(texts):
            if t_i > 0:
                predicted += ' '
            if t_i + 1 < len(texts) and self.try_merge(texts[t_i], texts[t_i + 1]):
                predicted += texts[t_i] + texts[t_i + 1]
                t_i += 2
            else:
                predicted += self.best_split(texts[t_i])
                t_i += 1
        predicted = self.postprocessor.correct(predicted)
        return predicted
