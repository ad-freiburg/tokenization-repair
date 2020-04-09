import unittest

from src.postprocessing.rule_based import NoSpaceBeforeRule, NoSpaceAfterRule, MergeDigitRule, QuotationRule, \
    HyphenatedWordRule, AlwaysMergeRule, RuleBasedPostprocessor


class RuleBasedPostprocessorTest(unittest.TestCase):
    def test_no_space_before_rule(self):
        rule = NoSpaceBeforeRule()
        self.assertTrue(rule.applies("bla", ",", ''))
        self.assertFalse(rule.applies("bla", "bla", ''))
        self.assertTrue(rule.applies("", ",", ''))
        self.assertFalse(rule.applies("", "bla", ''))
        self.assertTrue(rule.applies("bla", "'s", ''))
        self.assertFalse(rule.applies("bla", "'six", ''))

    def test_no_space_after_rule(self):
        rule = NoSpaceAfterRule()
        self.assertTrue(rule.applies("bla", "(", ''))

    def test_merge_digit_rule(self):
        rule = MergeDigitRule()
        self.assertTrue(rule.applies("1", "2", ''))
        self.assertFalse(rule.applies("1", ",", ''))
        self.assertFalse(rule.applies("bla", "1", "2"))

    def test_quotation_rule(self):
        rule = QuotationRule()
        self.assertTrue(rule.read_even_number)
        self.assertFalse(rule.merge_before())
        self.assertTrue(rule.merge_after())
        self.assertTrue(rule.applies("bla", "\"", ""))
        rule.apply("bla", "\"", "")
        self.assertFalse(rule.read_even_number)
        self.assertTrue(rule.merge_before())
        self.assertFalse(rule.merge_after())
        self.assertTrue(rule.applies("bla\"", "\"", ""))

    def test_hyphenated_word_rule(self):
        rule = HyphenatedWordRule()
        self.assertTrue(rule.applies("character", "-", "based"))
        self.assertFalse(rule.applies("9", "-", "5"))

    def test_always_merge_rule(self):
        rule = AlwaysMergeRule()
        self.assertTrue(rule.applies("bla", "\u2009", "bli"))

    def test_postprocessor(self):
        self.assertEqual("The cat eats fish.", RuleBasedPostprocessor.correct("The cat eats fish."))
        self.assertEqual("The cat, who likes fish.",
                         RuleBasedPostprocessor.correct("The cat , who likes fish ."))
        self.assertEqual("Bla's 'statement': \"bli.\"",
                         RuleBasedPostprocessor.correct("Bla 's 'statement' : \" bli . \""))
        self.assertEqual('""', RuleBasedPostprocessor.correct('" "'))
        self.assertEqual('bla "" bli', RuleBasedPostprocessor.correct('bla " " bli'))
        self.assertEqual("I ate 123 apples.", RuleBasedPostprocessor.correct("I ate 1 2 3 apples."))
        self.assertEqual("I use character-based language models.",
                         RuleBasedPostprocessor.correct("I use character - based language models ."))
        self.assertEqual("bla (bli) blu", RuleBasedPostprocessor.correct("bla ( bli ) blu"))
        self.assertEqual("()", RuleBasedPostprocessor.correct("( )"))
        self.assertEqual("bla () bli", RuleBasedPostprocessor.correct("bla ( ) bli"))

    def test_quotation_beginning(self):
        self.assertEqual("\"bla\"", RuleBasedPostprocessor.correct("\" bla \""))

    def test_combined_case(self):
        self.assertEqual("bla (\"bli\") blu", RuleBasedPostprocessor.correct("bla ( \" bli \" ) blu"))


if __name__ == "__main__":
    unittest.main()
