from src.ngram.unigram_holder import UnigramHolder
from src.ngram.bigram_holder import BigramHolder
from src.ngram.tokenizer import Tokenizer
from src.postprocessing.rule_based import RuleBasedPostprocessor
from src.fuzzy.matcher import FuzzyMatcher


class FuzzyGreedyCorrector:
    PENALTY = 0.1

    def __init__(self):
        unigrams = UnigramHolder()
        print("%i unigrams" % len(unigrams))
        bigrams = BigramHolder.load()
        print("%i bigrams" % len(bigrams))
        self.matcher = FuzzyMatcher(unigrams, bigrams, self.PENALTY)
        print("%i stumps" % len(self.matcher.stump_dict))
        self.tokenizer = Tokenizer()
        self.rule_based_postprocessor = RuleBasedPostprocessor()

    def correct(self, sequence: str):
        tokens = self.tokenizer.tokenize(sequence)
        texts = [token.text for token in tokens]
        predicted = ""
        t_i = 0
        while t_i < len(texts):
            if t_i > 0:
                predicted += ' '
            text = texts[t_i]
            if not text.isalpha():
                predicted += text
                t_i += 1
                continue
            # try merge:
            if t_i + 1 < len(texts) and texts[t_i + 1].isalpha():
                _, bigram_frequency = self.matcher.fuzzy_bigram_frequency(text, texts[t_i + 1])
                merge = text + texts[t_i + 1]
                _, merge_frequency = self.matcher.fuzzy_unigram_frequency(merge)
                if merge_frequency * self.PENALTY > bigram_frequency:
                    predicted += merge
                    t_i += 2
                    continue
            # try split:
            if len(text) > 1:
                _, unigram_frequency = self.matcher.fuzzy_unigram_frequency(text)
                split, _, split_frequency = self.matcher.best_fuzzy_split(text, lower_bound=unigram_frequency)
                if split_frequency * self.PENALTY > unigram_frequency:
                    predicted += ' '.join(split)
                    t_i += 1
                    continue
            predicted += text
            t_i += 1
        predicted = self.rule_based_postprocessor.correct(predicted)
        return predicted
