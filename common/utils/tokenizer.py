"""
Module containing text tokenizer class.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
from constants import ALPHABET, charset_of  #, TOKENIZATION_DELIMITERS
from .token import Token


class Tokenizer:
    """
    Tokenizer of text.
    """
    def __init__(self, text, dictionary, use_alphabet=False):
        """
        :param str text: The text to be tokenizer
        :param Dictionary dictionary: A dictionary of the words and idioms.
        """
        self.text = text
        self.dictionary = dictionary
        self.use_alpha = use_alphabet

    def __iter__(self):
        self.idx = 0
        self.tokens = []
        return self

    def is_token_char(self, char):
        return ((not self.use_alpha and char not in ' \t') or
                (self.use_alpha and char in ALPHABET))

    def __next__(self):
        if self.idx >= len(self.text):
            raise StopIteration
        st_token = self.idx
        charset = -1
        while (self.idx < len(self.text) and
               self.is_token_char(self.text[self.idx])):
            cset = charset_of(self.text[self.idx])
            if not cset == charset and not charset == -1:
                break
            charset = cset
            self.idx += 1
        en_token = st_splitter = self.idx
        while (self.idx < len(self.text) and
               not self.is_token_char(self.text[self.idx])):
            self.idx += 1
        en_splitter = self.idx

        tk = Token(self.text[st_token:en_token],
                   self.text[st_splitter:en_splitter])
        self.tokens.append(tk)

        return tk

    def get_all_tokens(self):
        """
        Construct all the tokens.

        :rtype: list(str)
        :returns: The list of processed tokens.
        """
        return self.iterate_all()[0]

    def iterate_all(self):
        """
        Construct all the tokens, formatted tokens and splitting tokens.

        :rtype: triple(list(str))
        :returns: A triple of the tokens, splitting tokens and fomatted tokens.
        """
        idx = 0
        tokens = []
        while not idx >= len(self.text):
            st_token = idx
            charset = -1
            while (idx < len(self.text) and
                   self.is_token_char(self.text[idx])):
                cset = charset_of(self.text[idx])
                if not cset == charset and not charset == -1:
                    break
                charset = cset
                idx += 1
            en_token = st_splitter = idx
            while (idx < len(self.text) and
                   not self.is_token_char(self.text[idx])):
                idx += 1
            en_splitter = idx

            tk = Token(self.text[st_token:en_token],
                       self.text[st_splitter:en_splitter])
            tokens.append(tk)

        return tokens

    def reconstruct_text(self):
        """
        Reconstruct the original text with the original tokens.

        :rtype: str
        :returns: The original text
        """
        tokens = self.iterate_all()
        text = str.join('', (token.totext() for token in tokens))
        return text

    def reconstruct_formatted_text(self):
        """
        Reconstruct the formatted text with the formatted tokens.

        :rtype: str
        :returns: The formatted text
        """
        tokens = self.iterate_all()
        text = str.join('', (token.totrimmedtext() for token in tokens))
        return text

    def reconstruct_fixed_text(self, fixed_tokens):
        """
        Reconstruct the original text with the fixed tokens.

        :param list(str) fixed_tokens:
            A list of words that correspond to the fixed tokens.
        :rtype: str
        :returns: The fixed text
        """
        text = str.join('', (token.totext() for token in fixed_tokens))
        return text
