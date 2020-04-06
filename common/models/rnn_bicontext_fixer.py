"""
Module containing bicontext text fixer using two directional RNN
character-based model.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
import multiprocessing
from configs import get_language_model_config
from .rnn_language_model import RNNLanguageModel
from constants import (
    DECODER_DICT,
    NUM_THREADS, EPS, MICRO_EPS, MODELS_ENUM,
    TYPO_ADD, TYPO_DEL, TYPO_NOCHANGE)
from utils.context_container import ContextContainer
from utils.logger import logger
from utils.utils import beam_search

import numpy as np


class RNNBicontextTextFixer:
    """
    RNN Bicontext fixer.
    """
    def __init__(self, config, from_tuner=False):
        self.forward_model_config = config.copy()
        self.forward_model_config.model = MODELS_ENUM.forward_language_model
        self.forward_model_config = get_language_model_config(
            **self.forward_model_config.__dict__)
        self.forward_language_model = RNNLanguageModel(self.forward_model_config)

        self.backward_model_config = config.copy()
        self.backward_model_config.model = MODELS_ENUM.backward_language_model
        self.backward_model_config = get_language_model_config(
            **self.backward_model_config.__dict__)
        self.backward_language_model = RNNLanguageModel(self.backward_model_config)

        self.beam_size = config.beam_size
        self.history_length = config.history_length
        self.alphabet = DECODER_DICT
        self.tokenization_delimiters = ' \t'
        self.fix_delimiters_only = config.fix_delimiters_only
        self.use_default_weights = config.use_default_weights
        self.use_look_forward = config.use_look_forward
        self.fixer_repr = config.fixer_repr

        if not from_tuner:
            logger.log_debug(self, ', beam:', self.beam_size, ', delimiters only:',
                             self.fix_delimiters_only, ', look forward:',
                             self.use_look_forward,
                             ', use default weights (no tuner):',
                             self.use_default_weights)
        if self.fix_delimiters_only or self.use_default_weights:
            """
            self.weights = [
                0.48471648, 1.2781795 , 0.6096173 , 0.74791324, 0.12425897,
                0.8286177 , 0.756555  , 0.76524204, 1.1021417 , 1.2689178]
            self.bias = [
                2.3979826 , -2.5865638 ,  3.5815628 ,  3.5622869 , 1.7156411 ,
                1.716309  ,  0.31246278,  0.31063208, -1.335301  , -1.3316967 ]
            """
            self.weights = [1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1]
            logF = np.log(0.005 + MICRO_EPS)
            self.bias = [0, logF, 0, 0, 0, 0, 0, 0, logF, logF]
        else:
            self.weights = [0.490227073431015, 0.395662784576416,
                            0.4301840662956238, 0.722004771232605,
                            0.2572476267814636, 0.6984100341796875,
                            0.7887752652168274, 0.824087917804718,
                            1.2120248079299927, 1.2566957473754883]
            self.bias = [1.0127992630004883, -4.189534664154053,
                         2.4342339038848877, 2.4149577617645264,
                         1.7975335121154785, 1.798201560974121,
                         0.7469261288642883, 0.7450954914093018,
                         -0.28035300970077515, -0.2767486572265625]
        if not self.use_default_weights:
            try:
                from .bicontext_tuner import BicontextTuner
                self.weights, self.bias = BicontextTuner(config).load_weights()
                logger.log_info('loaded weights..\n', self.weights, '\n', self.bias)

            except Exception as err:
                if not from_tuner:
                    logger.log_error(err)
                    logger.log_error('weights not found, please train tuner!.. '
                                     'using default weights')
                    logger.log_info('loaded temp weights..\n', self.weights, '\n', self.bias)
        else:
            logger.log_info('USING default weights..', self.weights, self.bias)

    def __repr__(self):
        """
        Short description of the model and the architecture used.

        :rtype: str
        :returns: short description of the model
        """
        return self.fixer_repr

    # {{{
    def predict_after_probability(
            self, after_context, before_context, small_context_siz=3,
            history=None):
        """
        Expectance that before_context is followed by after_context,
        according to forward_language_model (forward RNN)

        :param str after_context: String of the after context
        :param str before_context: String of the before context
        :param int small_context_siz: Considered long look length
        :rtype: float
        :return:
            Expected number of characters from after_context that comes
            after the before_context.
        """
        if small_context_siz <= 0:
            return 0
        after_context = after_context.restrict_backward(small_context_siz + 1)
        after_length = len(after_context.get_context())
        if after_length == 0:
            p = self.forward_language_model.predict_end(
                before_context.get_context())
            if history is not None:
                history.append(np.log(p + MICRO_EPS))
            return p
        first_after = after_context.poll_character()
        p = self.forward_language_model.predict_char(
            before_context.get_context(), first_after)

        if history is not None:
            history.append(np.log(p + MICRO_EPS))
        if after_length == 1:
            return p
        before_context = before_context.copy()
        before_context.append_context(first_after)
        return p * (1 + self.predict_after_probability(
            after_context, before_context,
            small_context_siz=small_context_siz - 1, history=history))

    def predict_before_probability(
            self, before_context, after_context, small_context_siz=3, history=None):
        """
        Expectance that after_context is preceded by before_context,
        according to backward_language_model (backward RNN)

        :param str before_context: String of the before context
        :param str after_context: String of the after context
        :param int small_context_siz: Considered long look length
        :rtype: float
        :return:
            Expected number of characters from before_context that comes
            before the after_context.
        """
        if small_context_siz <= 0:
            return 0
        before_context = before_context.restrict_afterward(
            small_context_siz + 1)
        before_length = len(before_context.get_context())
        if before_length == 0:
            p = self.backward_language_model.predict_start(
                after_context.get_context())
            if history is not None:
                history.insert(0, np.log(p + MICRO_EPS))
            return p
        last_before = before_context.pop_character()
        p = self.backward_language_model.predict_char(
            after_context.get_context(), last_before)
        if history is not None:
            history.insert(0, np.log(p + MICRO_EPS))
        if before_length == 1:
            return p
        after_context = after_context.copy()
        after_context.push_context(last_before)
        return p * (1 + self.predict_before_probability(
            before_context, after_context,
            small_context_siz=small_context_siz - 1, history=history))

    def predict_occurence(self, before_context, after_context):
        """
        Predict occurence

        :param str after_context: String of the after context
        :param str before_context: String of the before context
        :return:
            Expected number of characters from after_context that comes
            after the before_context.
        """
        return (self.predict_after_probability(after_context, before_context) *
                self.predict_before_probability(before_context, after_context))
    # }}}

    def get_actions_probabilities(self, before_context, after_context,
                                  current, correct_operation):
        """
        Generate probabilities vector of all possible actions of a given state,
        and the corresponding output vector of decisions.
        This will be used to feed input data for the tuner.

        :param str before_context: Context before the being fixed character
        :param str after_context: Context after the being fixed character
        :param str current: character being fixed
        :param ndarray correct_operation: correct operation
        :rtype: pair(vector, vector)
        :returns: Pair of probability and one-hot actions vectors
        """
        probs = np.zeros((10,))
        v = current

        X = before_context.copy()
        Xv = X.copy()
        Xv.append_context(v)
        Y = after_context.copy()
        Z = Y.copy()
        Z.push_context(v)
        # ###############################################################
        pr_del = self.predict_occurence(X, Y)
        if v in self.tokenization_delimiters or v == '-':
            probs[0] = pr_del
        else:
            probs[1] = pr_del
        # ###############################################################
        pr_nochg1 = self.predict_occurence(Xv, Y)
        pr_nochg2 = self.predict_occurence(X, Z)
        if v in self.tokenization_delimiters:
            probs[2] = pr_nochg1
            probs[3] = pr_nochg2
        else:
            probs[4] = pr_nochg1
            probs[5] = pr_nochg2

        # ###############################################################
        try_to_addF = self.forward_language_model.predict_top_n(
            X.get_context())
        try_to_addB = self.backward_language_model.predict_top_n(
            Z.get_context())
        try_to_add = set(try_to_addF) | set(try_to_addB) | {' '}
        pr_add1_nosp, to_add1 = 0, None
        pr_add2_nosp, to_add2 = 0, None
        pr_add1_sp = 0
        pr_add2_sp = 0
        for s in try_to_add:
            Xs = X.copy()
            Xs.append_context(s)

            sZ = Z.copy()
            sZ.push_context(s)

            pr_add1 = self.predict_occurence(X, sZ)
            pr_add2 = self.predict_occurence(Xs, Z)
            if s in self.tokenization_delimiters:
                pr_add1_sp = max(pr_add1_sp, pr_add1)
                pr_add2_sp = max(pr_add2_sp, pr_add2)
            else:
                if pr_add1 > pr_add1_nosp:
                    pr_add1_nosp = pr_add1
                    to_add1 = s
                if pr_add2 > pr_add2_nosp:
                    pr_add2_nosp = pr_add2
                    to_add2 = s
        if (isinstance(correct_operation, tuple) and
                correct_operation[0] == TYPO_ADD and
                correct_operation[1] not in self.tokenization_delimiters):
            if to_add1 != correct_operation[1]:
                # pr_add1_nosp = 0.0
                pr_add1_nosp = 1.0
            if to_add2 != correct_operation[1]:
                # pr_add2_nosp = 0.0
                pr_add2_nosp = 1.0
        probs[6] = pr_add1_sp
        probs[7] = pr_add2_sp
        probs[8] = pr_add1_nosp
        probs[9] = pr_add2_nosp
        # ###############################################################

        label = -1
        if (correct_operation == TYPO_DEL and
                (current in self.tokenization_delimiters or current == '-')):
            # delete delimiter
            label = 0
        elif (correct_operation == TYPO_DEL and
                current not in self.tokenization_delimiters and
                current != '-'):
            # delete char
            label = 1
        elif (correct_operation == TYPO_NOCHANGE and
                current in self.tokenization_delimiters):
            # keep delimiter
            label = 2
        elif (correct_operation == TYPO_NOCHANGE and
                current not in self.tokenization_delimiters):
            # keep char
            label = 3
        elif (isinstance(correct_operation, tuple) and
                correct_operation[0] == TYPO_ADD and
                correct_operation[1] in self.tokenization_delimiters):
            # add delimiter
            label = 4
        elif (isinstance(correct_operation, tuple) and
                correct_operation[0] == TYPO_ADD and
                correct_operation[1] not in self.tokenization_delimiters):
            # add char
            label = 5
        else:
            assert False, 'invalid operation %s' % str(correct_operation)

        return probs + MICRO_EPS, label

    def fix_step(self, text, score, before_context, after_context,
                 cur_pos, res_text, last_op, added=0):
        """
        Fix a given state of the text.

        :param str text: The given corrupt text
        :param float score: Overall fixing score so far
        :param str before_context: The context before the character to be fixed
        :param str after_context: The context after the character to be fixed
        :param int cur_pos: Position of the character being fixed in the text
        :param str res_text: The resulting fixed text so far
        :param last_op:
            Last used operation, used for debugging during beamsearch
        :param int added:
            The number of added characters since the last non-add operation
        :rtype: 7-tuple(float, str, str, int, str, operation, int)
        :returns: generator of (updated score, updated before_context,
                                updated after_context, new position,
                                updated res_text, used operation, new added)
        """
        before_context_copy = before_context.copy()
        after_context_copy = after_context.copy()
        if cur_pos < len(text):
            v = text[cur_pos]
            pos = cur_pos

            X = before_context.copy()
            Xv = X.copy()
            Xv.append_context(v)
            Y = after_context.copy()
            Z = Y.copy()
            Z.push_context(v)

            # ###############################################################
            if (not self.fix_delimiters_only or
                    v in self.tokenization_delimiters or v == '-'):
                pr_del = self.predict_occurence(X, Y)
                before_context = before_context_copy.copy()
                after_context = after_context_copy.copy()
                after_context.poll_character()
                after_context.append_context(
                    text[pos + self.history_length + 1:][:1])

                pr_del = np.log(pr_del + MICRO_EPS)
                if v in self.tokenization_delimiters or v == '-':
                    pr_del = pr_del * self.weights[0] + self.bias[0]
                else:
                    pr_del = pr_del * self.weights[1] + self.bias[1]
                yield (score - pr_del, before_context,
                       after_context, cur_pos + 1, res_text, TYPO_DEL, 0)
            # ###############################################################
            before_context = before_context_copy.copy()
            after_context = after_context_copy.copy()
            pr_nochg1 = self.predict_occurence(Xv, Y)
            pr_nochg2 = self.predict_occurence(X, Z)
            before_context = before_context_copy.copy()
            after_context = after_context_copy.copy()
            to_add = v
            before_context.append_context(to_add)
            after_context.poll_character()
            after_context.append_context(
                text[pos + self.history_length + 1:][:1])
            pr_nochg1 = np.log(pr_nochg1 + MICRO_EPS)
            pr_nochg2 = np.log(pr_nochg2 + MICRO_EPS)

            if v in self.tokenization_delimiters:
                pr_nochg1 = pr_nochg1 * self.weights[2] + self.bias[2]
                pr_nochg2 = pr_nochg2 * self.weights[3] + self.bias[3]
            else:
                pr_nochg1 = pr_nochg1 * self.weights[4] + self.bias[4]
                pr_nochg2 = pr_nochg2 * self.weights[5] + self.bias[5]
            pr_nochg = pr_nochg1 + pr_nochg2

            yield (score - pr_nochg, before_context,
                   after_context, cur_pos + 1, res_text + to_add,
                   TYPO_NOCHANGE, 0)

            # ###############################################################

            try_to_add = {' '}
            if not self.fix_delimiters_only:
                try_to_addF = self.forward_language_model.predict_top_n(
                    X.get_context())
                try_to_addB = self.backward_language_model.predict_top_n(
                    Z.get_context())
                try_to_add = set(try_to_addF) | set(try_to_addB) | {' '}
            for s in try_to_add:
                Xs = X.copy()
                Xs.append_context(s)

                sY = Y.copy()
                sY.push_context(s)

                sZ = Z.copy()
                sZ.push_context(s)

                pr_add1 = self.predict_occurence(X, sZ)
                pr_add2 = self.predict_occurence(Xs, Z)

                pr_add1 = np.log(pr_add1 + MICRO_EPS)
                pr_add2 = np.log(pr_add2 + MICRO_EPS)
                if s in self.tokenization_delimiters:
                    pr_add1 = pr_add1 * self.weights[6] + self.bias[6]
                    pr_add2 = pr_add2 * self.weights[7] + self.bias[7]
                else:
                    pr_add1 = pr_add1 * self.weights[8] + self.bias[8]
                    pr_add2 = pr_add2 * self.weights[9] + self.bias[9]
                pr_add = pr_add1 + pr_add2
                pr_add_forward = np.log(MICRO_EPS)
                if pr_add > -30 and self.use_look_forward:
                    csZ = Z.copy()
                    cs = ContextContainer(csZ.history_siz)
                    for _ in Z.get_context()[0:2]:
                        cs.append_context(csZ.poll_character())
                        csZ.push_context(s)
                        rsZ = csZ.copy()
                        rsZ.push_context(cs.get_context())
                        pr_add_forward = max(
                            pr_add_forward,
                            np.log(self.predict_occurence(X, rsZ) + MICRO_EPS))

                        if pr_add_forward > pr_add:
                            break
                        assert s == csZ.poll_character()

                if ((pr_add_forward > pr_add or pr_add < -30) and
                        self.use_look_forward):
                    pr_add = np.log(MICRO_EPS)

                if added <= 2:
                    before_context = before_context_copy.copy()
                    after_context = after_context_copy.copy()
                    before_context.append_context(s)
                    yield (score - pr_add, before_context,
                           after_context, cur_pos, res_text + s,
                           (TYPO_ADD, s), added + 1)

    def fix(self, text):
        """
        Fix a given text using beam search, given the implementation of
        fix_step provides the next state in the search.

        The state is on the form:
            (total_score, before_context container, after_context container,
             current_pos, result_text, last_operation, last_added)

        :param str text: Given corrupt text
        :rtype: str
        :returns: Fixed version of the text
        """
        def is_terminal_state(state):
            """
            Local function to determine if a given state is terminal,
            in order to be used by beam search.

            :param tuple state: A state of the beam search
            :rtype: boolean
            :returns: boolean if the given state is terminal
            """
            (score, bef_context, after_context, pos,
             res_text, last_op, added) = state
            return pos >= len(text)

        def next_states(state):
            """
            Local function to enumerate all the next states of a given state.

            :param tuple state: A state of the beam search
            :rtype: generator(state)
            :returns: an enumeration of all successor states of the given state
            """
            for next_state in self.fix_step(text, *state):
                yield next_state

        before_context = ContextContainer(self.history_length)
        after_context = ContextContainer(self.history_length,
                                         text[1: 1 + self.history_length])
        logger.log_full_debug('total', len(text))
        logger.log_full_debug(text)
        fixation = beam_search((0.0, before_context, after_context,
                                0, '', 'beg', 0),
                               is_terminal_state, next_states,
                               # Compare the functions by costs: state[0]
                               (lambda x: x[0]),
                               self.beam_size)
        logger.log_full_debug(fixation)
        score, _bef, _aft, _pos, fixed_text, last_op, _ = fixation
        return fixed_text
