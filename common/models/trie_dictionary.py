"""
Module containing dictionary class, maintining dictionary of words using
trie data structure.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
import math

from configs import default_trie_dictionary_config
from util.constants import EPS, SPECIAL
from util.logger import logger
from util.token import Token


class TrieDictionary:
    """
    Dictionary object to hold the dictionary of a language.
    """
    def __init__(self, config=default_trie_dictionary_config):
        """
        Construct a dictionary.

        :param list(str) words: List of the words of the dictionary.
        :param list(str) idioms: List of the idioms of the dictionary.
        """
        self.nexts = {0: {}}
        self.marked = {}
        self.counts = {}
        self.freqs = {}

        self.parent = {}
        self.marked_children = {}
        self.cumulative_freq = {}
        self.size = 0
        self.total = 0
        self.insert(('', 1))
        self.edit_damping_factor = config.damping_factor
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.zeta = config.zeta
        for path in config.dictionary_paths:
            logger.log_info("Loading Trie dictionary from: ", path)
            self.load_words_from_file(path)

    def construct_parents(self, node=0):
        self.marked_children[node] = 0
        self.cumulative_freq[node] = 0
        for char, nxt in self.nexts[node].items():
            self.construct_parents(nxt)
            self.parent[nxt][char] = node
            self.marked_children[node] += self.marked_children[nxt]
            self.cumulative_freq[node] += self.cumulative_freq[nxt]
        if node in self.marked.keys():
            self.marked_children[node] += 1
            self.cumulative_freq[node] += self.freqs[node]

    def get_terminal_states(self):
        """
        Get terminal states in the trie.

        :rtype: list
        :returns: list of marked or terminal nodes
        """
        return list(self.marked.keys())

    def get_parent(self, state, char):
        """
        Get parent state in a given Trie, using a character

        :param int state: Given node
        :param char char: character to use to go to parent
        :rtype: int
        :returns: parent node index
        """
        if char in self.parent[state].keys():
            return self.parent[state][char]
        return None

    def get_all_parents(self, state):
        """
        Get parent state in a given Trie, using a character

        :param int state: Given node
        :rtype: list(int)
        :returns: parent node index in a list
        """
        return list(self.parent[state].items())

    def is_root(self, state):
        """
        Check if a node is a root state

        :rtype: bool
        :returns: True if a node is root
        """
        return state == 0

    def is_terminal_state(self, state):
        """
        Check if a given state is marked (terminal)

        :param int state: Index of the node
        :rtype: bool
        :returns: True if the given state is marked
        """
        return state in self.marked.keys()

    def next_state(self, state, char):
        """
        Get the next state of a node using a char.

        :param int state: Given node
        :param char char: Given character to use
        :rtype: int
        :returns: The next state using state and char
        """
        if state in self.nexts.keys() and char in self.nexts[state].keys():
            return self.nexts[state][char]

    def next_states(self, state):
        """
        Get all the next states of a given state

        :param int state: Node index
        :rtype: list(pair)
        :returns: list of pairs of characters and corresponding next states.
        """
        if state in self.nexts.keys():
            return self.nexts[state].items()
        return []

    def get_state_freq(self, state):
        """
        Get the frequency of a given node

        :param int state: Node index
        :rtype: int
        :returns: Frequency of a given state
        """
        if state in self.freqs.keys():
            return self.freqs[state]
        return 0

    def add_words(self, vocab):
        """
        Add words to the dictionary.

        :param list(str) vocab: List of words
        """
        for word in vocab:
            if '\t' in word:
                word, freq = word.split('\t')
                freq = int(freq)
            else:
                freq = 16
            if freq > 15:
                if len(word) > 2 and word[1:].lower() == word[1:]:
                    self.insert((word.lower(), freq))
                else:
                    self.insert((word, freq))

    def load_words_from_file(self, file_path):
        """
        Load wordsfrom a file.

        :param str file_path: A string of the path of the file.
        """
        with open(file_path, 'r') as words_file:
            words = words_file.read().split('\n')
            self.add_words(words)

    def load_idioms_from_file(self, file_path):
        """
        Load idioms from a file.

        :param str file_path: A string of the path of the file.
        """
        with open(file_path, 'r') as idioms_file:
            idioms = idioms_file.read().split('\n')
            for idiom in idioms:
                self.insert(idiom.lower())

    def density(self, length):
        """
        Return the density of words of a given length.

        :param int length:
        :rtype: float
        """
        return ((1.0 + sum(count for key, count in self.counts.items()
                           if key <= length)) / float(self.size + 1))
        # if length in self.counts.keys():
        #     return (1 + self.counts[length]) / float(self.size + 1)
        # else:
        #     return 1 / float(self.size + 1)

    def new_node(self):
        """
        Create a new node

        :rtype: int
        :returns: the new node
        """
        idx = len(self.nexts)
        self.nexts[idx] = {}
        self.parent[idx] = {}
        return idx

    def get_or_create_next(self, root, key):
        """
        Get or create (if doesn't exist) the next of (root, key)

        :param int root: root node
        :param char key: character
        :rtype: int
        :returns: the next node
        """
        if root in self.nexts.keys():
            if key in self.nexts[root].keys():
                return self.nexts[root][key]
            else:
                self.nexts[root][key] = self.new_node()
                return self.nexts[root][key]
        else:
            self.nexts[root] = {}
            self.nexts[root][key] = self.new_node()
            return self.nexts[root][key]

    def insert(self, word, idx=0, root=0):
        """
        Insert a word into dictionary.

        :param str word: The new word
        """
        word, freq = word
        if idx == 0:
            self.all_words[word] = self.all_words.get(word, 0) + 1
        if len(word) > 1 or word == 'a':
            if idx >= len(word):
                self.marked[root] = len(word)
                if root not in self.freqs.keys():
                    self.freqs[root] = 0
                self.freqs[root] += freq
                if len(word) not in self.counts.keys():
                    self.counts[len(word)] = 0
                self.counts[len(word)] += 1
                self.size += 1
                self.total += freq
            else:
                self.insert((word, freq), idx + 1,
                            self.get_or_create_next(root, word[idx]))

    def exists(self, word, idx=0, root=0, edit_dist=0):
        """
        Check if an expression exists in dictionary.

        :param str expr: The query expression
        :returns: True if the expression exists
        :rtype: boolean
        """
        score = 0
        typo = ''
        if idx >= len(word):
            if root in self.marked:
                score, ntypo = 1, ''
                # logger.log_full_debug(
                #     word, ntypo, self.freqs[root] / float(self.total),
                #     math.pow(self.freqs[root] / float(self.total), 0.5))
                score *= math.pow(self.freqs[root] / float(self.total), 0.05)
            if edit_dist > 0 and score < EPS:
                for c, nx in self.nexts[root].items():
                    if "'" in c:
                        continue
                    # Add c before word[idx]
                    score, ntypo = self.exists(word, idx, nx, edit_dist - 1)
                    p = (len(ntypo) + 1) / float(len(ntypo) + 2)
                    p = 1.0
                    score *= self.edit_damping_factor * p
                    if not score < EPS:
                        typo = c + ntypo
                        break
        else:
            assert root in self.nexts.keys()

            if idx == 0 and len(word) > 2 and word[1:].lower() == word[1:]:
                ci = word[idx].lower()
            else:
                ci = word[idx]

            if ci in self.nexts[root].keys():
                score, ntypo = self.exists(word, idx + 1,
                                           self.nexts[root][ci],
                                           edit_dist)
                if not score < EPS:
                    typo = word[idx] + ntypo
            if edit_dist > 0 and score < EPS:
                if score < EPS:
                    # Delete word[idx]
                    score, ntypo = self.exists(word, idx + 1, root,
                                               edit_dist - 1)
                    p = (len(ntypo) + 1) / float(len(ntypo) + 2)
                    p = 1.0
                    score *= self.edit_damping_factor * p
                    typo = '' + ntypo
                if score < EPS:
                    for c, nx in self.nexts[root].items():
                        if c == "'":
                            continue
                        assert "'" not in c
                        # Substitute word[idx] by c
                        score, ntypo = self.exists(word, idx + 1, nx,
                                                   edit_dist - 1)
                        p = (len(ntypo) + 1) / float(len(ntypo) + 2)
                        p = 1.0
                        score *= self.edit_damping_factor * p
                        if not score < EPS:
                            typo = c + ntypo
                            break
                        # Add c before word[idx]
                        score, ntypo = self.exists(word, idx, nx,
                                                   edit_dist - 1)
                        p = (len(ntypo) + 1) / float(len(ntypo) + 2)
                        p = 1.0
                        score *= self.edit_damping_factor * p
                        if not score < EPS:
                            typo = c + ntypo
                            break
        return score, typo

    def word_exists(self, word):
        """
        Check if a word exists in dictionary.

        :param str word: The query word
        :returns: True if the word exists
        :rtype: boolean
        """
        return word in self.words

    def join_tokens(self, tokens):
        """
        Join a list of tokens.

        :param list(str) tokens: The list of tokens.
        :rtype: str
        :returns: A string of the combined token.
        """
        last_good_split = list(filter(
            lambda tk: not (tk.get_word().endswith('-') and
                            tk.get_split() == '\n'), tokens))
        if not last_good_split:
            last_good_split = ' '
        else:
            last_good_split = last_good_split[-1].get_split()
        ret = Token(''.join(self.trim(token.get_word()) for token in tokens
                            if token.get_word() != '-'),
                    last_good_split)
        return ret

    def token_score(self, token):
        """
        Compute a matching score of a token.

        :param str token: query word.
        :rtype: float
        :returns: Score of the token, higher means better matches dictionary
        """
        if token.word.endswith('-') and token.split == '\n':
            token = Token(token.word[:-1], '')
        if ((len(token.get_word()) <= 4 and len(token.get_word()) >= 3 and
             token.get_word() == token.get_word().upper()) or
                all(c in SPECIAL for c in token.get_word())):
            return self.zeta, token
        score, typo = self.exists(self.trim(token.get_word()),
                                  edit_dist=1)

        if not typo:
            typo = token.get_word()
        w = len(token.get_word())
        # if score < 1.0:
        #     score *= score  # self.density(len(typo)) * score
        # return (score * ((len(token.get_word()) / 15.0) ** 2 + 0.05),
        #         Token(typo, token.get_split()))
        return (score * (w * w * self.alpha + w * self.beta + self.gamma),
                Token(typo, token.get_split()))

    def trim(self, word):
        """
        Trim a given word.

        :param str word: The given word.
        :rtype: str
        :returns: The trimmed word.

        >>> d = Dictionary()
        >>> d.trim('hello')
        'hello'
        >>> d.trim('hello-')
        'hello'
        >>> d.trim('hello--')
        'hello'
        """
        if len(word) > 1 and word[-1] == '-':
            return self.trim(word[:-1])
        return word
