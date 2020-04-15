from typing import List

from nltk.tokenize import word_tokenize


class Token:
    def __init__(self, text: str, delimiter_before: bool):
        self.text = text
        self.delimiter_before = delimiter_before

    def __str__(self):
        return "Token(%s, %s)" % (self.text, str(self.delimiter_before))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.delimiter_before == other.delimiter_before and self.text == other.text


def tokens2sequence(tokens: List[Token]) -> str:
    text_snippets = []
    for t_i, token in enumerate(tokens):
        if t_i > 0 and token.delimiter_before:
            text_snippets.append(' ')
        text_snippets.append(token.text)
    sequence = ''.join(text_snippets)
    return sequence


class Tokenizer:
    @staticmethod
    def _make_tokens(snippets: List[str], sequence) -> List[Token]:
        thin_space = '\u2009'
        tokens = []
        pos = 0
        for t_i, snippet in enumerate(snippets):
            delimiter = False

            if sequence[pos] == thin_space:
                tokens.append(Token(thin_space, False))
                pos += 1
            if sequence[pos] == ' ':
                delimiter = True
                pos += 1
            if sequence[pos] == thin_space:
                tokens.append(Token(thin_space, delimiter))
                delimiter = False
                pos += 1

            if delimiter and sequence[pos] == ' ':
                tokens.append(Token('', delimiter))
                pos += 1

            if sequence[pos] == '"':
                text = '"'
            elif snippet == "``" and sequence[pos:(pos + 2)] == "''":
                text = "''"
            else:
                text = snippet

            tokens.append(Token(text, delimiter))

            pos += len(text)
        return tokens

    @staticmethod
    def tokenize(sequence: str) -> List[Token]:
        snippets = word_tokenize(sequence)
        tokens = Tokenizer._make_tokens(snippets, sequence)
        return tokens
