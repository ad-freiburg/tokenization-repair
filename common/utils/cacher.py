"""
Module containing utility functions for viewing the output.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""


def hashstate(state):
    return tuple([x.tobytes() for x in state])


class Debugger:
    def __init__(self):
        self.cacher = {}

    def insert(self, state, total_string):
        self.cacher[hashstate(state)] = total_string

    def get(self, state):
        return self.cacher.get(hashstate(state), '')


class Cacher:
    """
    Cacher data-structure for some function
    """
    def __init__(self, cache_siz):
        self.cache_siz = cache_siz
        self.cache_dict = {}
        self.ids_dict = {}
        self.history = []

    def add_cache_value(self, key, value):
        """
        Add a key, value pair

        :param key: key of the function
        :param value: Value of the function for the given key
        """
        self.cache_dict[key] = value
        new_id = len(self.ids_dict)
        self.ids_dict[key] = new_id
        self.history.append(key)
        self._clear_old()

    def get_cached_value(self, key):
        """
        Get the cached value of some key, or None

        :param key: Key of the function
        :returns:
            The value corresponding to the cached key,
            or None if it doesn't exist
        """
        if key in self.cache_dict.keys():
            val = self.cache_dict[key]
            self.add_cache_value(key, val)
            return val

    def _clear_old(self):
        if len(self.history) > 2 * self.cache_siz:
            for old_key in self.history[: self.cache_siz]:
                if (old_key in self.cache_dict.keys() and
                        self.ids_dict[old_key] < self.cache_siz):
                    del self.cache_dict[old_key]
            self.history = self.history[-self.cache_siz:]
            self.ids_dict = {}
            for id, key in enumerate(self.history):
                self.ids_dict[key] = id
