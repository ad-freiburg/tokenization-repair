"""
Module containing the baseline tokens fixer based on dynamic programming.
"""
import _pickle as pickle

from .trie_dictionary import TrieDictionary
from constants import EPS, NEG_INFINITY, SPLIT_ENUM, USE_WHOLE_ENUM
from utils.logger import logger
from utils.token import Token
from utils.tokenizer import Tokenizer
from utils.utils import findmax


class DPFixer:
    """
    Text fixer using the dynamic programming approach on tokens.
    """
    def __init__(self, config):
        self.window_siz = config.window_siz
        self.fixer_repr = config.fixer_repr
        with open(config.dictionary_path, 'rb') as fl:
            self.dictionary = pickle.load(fl)
            logger.log_info('loaded dictionary from:', config.dictionary_path)

    def __repr__(self):
        return self.fixer_repr

    def retokenise(self, token):
        """
        Attempt to split a given token into good tokens.

        :param Token token: The query token
        :rtype: pair(int, list(Token))
        :returns: A pair of the matching score and splitted portions
        """
        # token = self.dictionary.join_tokens(self.tokens[st: st + window])
        big_word = token.get_word()
        # logger.log_full_debug(big_word)
        F = [NEG_INFINITY for _ in range(len(big_word) + 1)]
        nx = [idx + 1 for idx in range(len(big_word) + 1)]
        F[len(big_word)] = 0
        for i in reversed(range(len(big_word))):
            F[i], nx[i] = \
                findmax((
                    ((self.dictionary.token_score(Token(big_word[i:j]))[0] *
                      (j - i) + (len(big_word) - j) * F[j]) /
                     (len(big_word) - i))
                    for j in range(i + 1, min(len(big_word), i + 25) + 1)),
                    NEG_INFINITY)
            nx[i] += i + 1
        res = []
        idx = 0
        while idx < len(big_word):
            _, tok = self.dictionary.token_score(Token(big_word[idx:nx[idx]],
                                                       ' '))
            if nx[idx] >= len(big_word):
                res.append(Token(tok.get_word(), token.get_split()))
                # logger.log_full_debug(token)
            else:
                res.append(Token(tok.get_word(), ' '))
            idx = nx[idx]

        logger.log_full_debug('retokenized', token, 'into', res)
        logger.log_full_debug('retokenization score:', F)
        logger.log_full_debug('retokenization nexts:', nx)
        # logger.log_full_debug([self.dictionary.token_score(x) for x in res])
        return F[0], res

    def try_to_fix(self, st, window):
        """
        A dynamic programming algorithm using memoization, that tries to find
        the best merging of tokens that maximizes the number of correct tokens
        within a given range.

        :param int st: Start index of tokens' range without offset
        :param int en: End index of tokens' range without offset
        :param int[][] table: The memoization table of the recursion
        :param int[][] split: The memoization table of the split indices
        :param int offset: The offset of the range of tokens
        :rtype: int
        :returns: The number of maximum correct tokens
        """
        assert window > 0
        # if table[st][window] is not None:
        #     return table[st][window][0]
        joined = self.dictionary.join_tokens(self.tokens[st: st + window])
        # return self.retokenise(joined)[0], SPLIT_ENUM
        score, _ = self.dictionary.token_score(joined)
        split_whole = False
        if score < EPS:
            score, _ = self.retokenise(joined)
            if score > EPS:
                split_whole = True
        if split_whole:
            split_whole = SPLIT_ENUM
        else:
            split_whole = USE_WHOLE_ENUM
        if logger.is_full_debug():
            if Token('-', '\n') in self.tokens[st: st + window]:

                logger.log_full_debug('from', st + 1, 'to', window,
                                      self.tokens[st: st + window], score,
                                      split_whole, joined, highlight=2)
            else:
                logger.log_full_debug('from', st + 1, 'to', window,
                                      self.tokens[st: st + window], score,
                                      split_whole, joined, highlight=2)
        return score, split_whole
        # table[st][window] = (score, split_whole)
        # return table[st][window][0]

    def retrieve_all(self, st, window, action):
        """
        Retrieve all the maximal fixed tokens.

        :param int st: Start index of tokens' range without offset
        :param int en: End index of tokens' range without offset
        :param int[][] split: The memoization table of the split indices
        :param int offset: The offset of the range of tokens
        :rtype: int
        :returns: The number of maximum correct tokens
        """
        res = []
        if action == SPLIT_ENUM:
            joined = self.dictionary.join_tokens(self.tokens[st: st + window])
            _, sp = self.retokenise(joined)
            res.extend(sp)
            if logger.is_full_debug():
                logger.log_full_debug("splitted",
                                      '$'.join(map(lambda tk: tk.get_word(),
                                                   self.tokens[st:
                                                               st + window])),
                                      list(map(lambda tk: self.dictionary.
                                               token_score(tk)[0],
                                               self.tokens[st:
                                                           st + window])),
                                      "into", sp, _,
                                      self.dictionary.token_score(joined))
        elif action == USE_WHOLE_ENUM:
            joined = self.dictionary.join_tokens(self.tokens[st:
                                                             st + window])
            _, tok = self.dictionary.token_score(joined)
            res.append(tok)
            if logger.is_full_debug():
                logger.log_full_debug("merged",
                                      '$'.join(map(lambda tk: tk.get_word(),
                                                   self.tokens[st:
                                                               st + window])),
                                      "into", tok, _)
        else:
            assert False
            # res.extend(self.retrieve_all(st, sp, split, offset))
            # res.extend(self.retrieve_all(sp, en, split, offset))
        return res

    def fix(self, text):
        """
        Fix a given text

        :param str text: Given text
        :rtype: list(Token)
        :returns: Fixed tokens
        """
        tokenizer = Tokenizer(text, self.dictionary, use_alphabet=False)
        self.tokens = list(tokenizer)
        if logger.is_full_debug():
            for token in self.tokens:
                logger.log_full_debug(token.word, '::::', token.split)
        F = [NEG_INFINITY for _ in range(len(self.tokens) + 1)]
        # best = [[] for _ in range(len(self.tokens) + 1)]
        nxt = [None for idx in range(len(self.tokens) + 1)]
        F[len(self.tokens)] = 0
        for idx in reversed(range(len(self.tokens))):
            for window_siz in range(1, 1 + min(self.window_siz,
                                               len(self.tokens) - idx)):
                score, action = self.try_to_fix(idx, window_siz)
                merge_attempt = F[idx + window_siz] + score
                if F[idx] <= merge_attempt:
                    F[idx] = merge_attempt
                    nxt[idx] = (idx + window_siz, action)
        if logger.is_full_debug():
            logger.log_full_debug('grouping scores:', F)
            logger.log_full_debug('nexts:', nxt)
        result = []
        idx = 0
        while idx < len(self.tokens):
            nxidx, action = nxt[idx]
            result.extend(self.retrieve_all(idx, nxidx - idx, action))
            idx = nxidx
        return tokenizer.reconstruct_fixed_text(result)

    def new_memoization_table(self):
        """
        Create new empty memoization table.

        :returns: A 2d table of None for handling ranges.
        :rtype: int[][]
        """
        return [[None for j in range(i + 1, self.window_siz + 1)]
                for i in range(self.window_siz)]
