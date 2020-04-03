from project import src
from src.tokenizer.dictionary_tokenization_corrector import DictionaryTokenizationCorrector
from src.tokenizer.tokenization_rules import RuleBasedTokenizer
from src.sequence.transformation import revert_corruptions


class BaselineTokenizer:
    def __init__(self, n=1000000, use_dictionary=True, use_rules=True, use_aspell=True, postprocessing_method=None):
        self.use_dictionary = use_dictionary
        if use_dictionary:
            self.dictionary_corrector = DictionaryTokenizationCorrector(n, use_aspell=use_aspell,
                                                                        postprocessing_method=postprocessing_method)
        self.use_rules = use_rules
        if use_rules:
            self.rule_based_corrector = RuleBasedTokenizer()

    def predict(self, sequence):
        if self.use_dictionary:
            dictionary_predictions, _ = self.dictionary_corrector.predict(sequence)
        else:
            dictionary_predictions = []
        if self.use_rules:
            rule_based_predictions = self.rule_based_corrector.predict(sequence)
        else:
            rule_based_predictions = []
        predictions = list(dictionary_predictions) + list(rule_based_predictions)
        predicted = revert_corruptions(sequence, [p[1] for p in predictions])
        return predictions, predicted
