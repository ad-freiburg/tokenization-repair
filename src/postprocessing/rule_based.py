import abc
from typing import List, Tuple


NO_SPACE_BEFORE = set(".,)]?!;:'").union({"'s"})
NO_SPACE_AFTER = set("([")
NO_SPACE_ANYWHERE = set("\u2009")


class Rule(abc.ABC):
    def __init__(self):
        pass

    def merge_before(self):
        return False

    def merge_after(self):
        return False

    @abc.abstractmethod
    def applies(self, hitherto: str, snippet: str, next: str) -> bool:
        raise NotImplementedError

    def side_effects(self):
        pass

    def apply(self, hitherto: str, snippet: str, next: str) -> Tuple[bool, bool]:
        merge_before, merge_after = False, False
        if self.applies(hitherto, snippet, next):
            merge_before, merge_after = self.merge_before(), self.merge_after()
            self.side_effects()
        return merge_before, merge_after


class MergeBeforeRule(Rule, abc.ABC):
    def merge_before(self):
        return True


class MergeAfterRule(Rule, abc.ABC):
    def merge_after(self):
        return True


class NoSpaceBeforeRule(MergeBeforeRule):
    def applies(self, hitherto: str, snippet: str, next: str) -> bool:
        return snippet in NO_SPACE_BEFORE


class NoSpaceAfterRule(MergeAfterRule):
    def applies(self, hitherto: str, snippet: str, next: str) -> bool:
        return snippet in NO_SPACE_AFTER


class MergeDigitRule(MergeBeforeRule):
    def applies(self, hitherto: str, snippet: str, next: str) -> bool:
        return len(hitherto) > 0 and hitherto[-1].isnumeric() and snippet.isnumeric()


class QuotationRule(Rule):
    def __init__(self):
        super().__init__()
        self.read_even_number = True

    def merge_after(self):
        return self.read_even_number

    def merge_before(self):
        return not self.read_even_number

    def side_effects(self):
        self.read_even_number = not self.read_even_number

    def applies(self, hitherto: str, snippet: str, next: str) -> bool:
        return snippet == '"'


class HyphenatedWordRule(MergeBeforeRule, MergeAfterRule):
    def applies(self, hitherto: str, snippet: str, next: str) -> bool:
        return snippet == '-' and len(hitherto) > 0 and hitherto[-1].isalpha() and next is not None \
               and next[0].isalpha()


class AlwaysMergeRule(MergeBeforeRule, MergeAfterRule):
    def applies(self, hitherto: str, snippet: str, next: str) -> bool:
        return snippet in NO_SPACE_ANYWHERE


class RuleBasedPostprocessor:
    RULES = [NoSpaceBeforeRule,
             NoSpaceAfterRule,
             MergeDigitRule,
             QuotationRule,
             HyphenatedWordRule,
             AlwaysMergeRule]

    @staticmethod
    def correct(sequence: str) -> str:
        rules = [rule() for rule in RuleBasedPostprocessor.RULES]
        snippets = sequence.split(' ')
        merge_before = False
        merge_after = False
        postprocessed_snippets = []
        for i, snippet in enumerate(snippets):
            for rule in rules:
                mb, ma = rule.apply(postprocessed_snippets[-1] if len(postprocessed_snippets) > 0 else '',
                                    snippets[i],
                                    snippets[i + 1] if i + 1 < len(snippets) else None)
                merge_before = merge_before or mb
                merge_after = merge_after or ma
            if i > 0 and merge_before:
                postprocessed_snippets[-1] += snippet
            else:
                postprocessed_snippets.append(snippet)
            merge_before = merge_after
            merge_after = False
        postprocessed = ' '.join(postprocessed_snippets)
        return postprocessed
