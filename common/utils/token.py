"""
Module containing a Token object.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""


class Token:
    def __init__(self, word, split=''):
        self.word = word
        self.split = split
        self.trimmed_word = self.format(word)

    def __eq__(self, other):
        if isinstance(other, Token):
            return other.word == self.word and other.split == self.split
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '%s%s' % (self.word, self.split)

    def get_word(self):
        return self.word

    def get_split(self):
        return self.split

    def totext(self):
        return '%s%s' % (self.word, self.split)

    def totrimmedtext(self):
        return '%s%s' % (self.trimmed_word, self.split)

    def format(self, word):
        """
        Format a given word.

        :param str word: The given word
        :returns: The formatted word
        :rtype: str
        """
        if not word:
            return word
        st = next((idx for idx, c in enumerate(word)
                   if c not in ' ') or len(word))
        en = next((len(word) - idx for idx, c in enumerate(reversed(word))
                   if c not in ' ') or len(word))
        return word[st:en]
