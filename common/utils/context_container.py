"""
Module containing a data-structure for maintaining a window of context.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
import numpy as np


class ContextContainer:
    """
    Container of fixed size context strings.
    """
    def __init__(self, history_siz, context=''):
        self.history_siz = history_siz
        self.context = np.string_('')
        self.append_context(context)

    def __repr__(self):
        return '%25s' % ('$' + self.get_context().replace('\n', ' ') + '$')

    def push_context(self, context):
        """
        Push a new context from the front

        :param str context: New context
        :returns: self
        """
        context = np.string_(context[:self.history_siz])
        if len(context) >= self.history_siz:
            self.context = context
        else:
            self.context = context + self.context
        self.context = self.context[:self.history_siz]
        return self

    def pop_character(self):
        """
        Pop the last character in the context

        :rtype: char
        :returns: The last character in context or None if it's empty
        """
        res = chr(self.context[-1]) if self.context else None
        self.context = self.context[:-1]
        return res

    def append_context(self, context):
        """
        Append a new context

        :param str context: New context
        :rtype: ContextContainer
        :returns: self

        >>> ContextContainer(6, 'hello').append_context('wor').get_context()
        'llowor'
        """
        context = np.string_(context[-self.history_siz:])
        if len(context) >= self.history_siz:
            self.context = context
        else:
            self.context = self.context + context
        self.context = self.context[-self.history_siz:]
        return self

    def poll_character(self):
        """
        Pop the first character in the context

        :rtype: char
        :returns: The first character in context or None if it's empty
        """
        res = chr(self.context[0]) if self.context else None
        self.context = self.context[1:]
        return res

    def restrict_backward(self, context_length):
        """
        Adjust context from backwards (take first n characters)

        :param int context_length: Number of characters to be taken
        :rtype: ContextContainer
        :returns: Copy of a Context containing first n characters
        """
        return ContextContainer(context_length, self.context[:context_length])

    def restrict_afterward(self, context_length):
        """
        Adjust context from forwards (take last n characters)

        :param int context_length: Number of characters to be taken
        :rtype: ContextContainer
        :returns: Copy of a Context containing last n characters
        """
        return ContextContainer(context_length, self.context[-context_length:])

    def get_context(self):
        """
        Get the stored context

        :rtype: str
        :returns: The contained context
        """
        return self.context.decode()

    def copy(self):
        """
        Create a copy of the object

        :rtype: ContextContainer
        :returns: A copy of the object
        """
        return ContextContainer(self.history_siz, self.context)
