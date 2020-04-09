"""
Module containing bicontext text fixer using two directional RNN
character-based model.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
import multiprocessing
import os
import warnings
import _pickle as pickle
from contextlib import closing
from datetime import datetime
from tqdm import tqdm

from configs import get_language_model_config
from .rnn_language_model import RNNLanguageModel, DEBUG_MODE
from constants import (
    DECODER_DICT, NUM_THREADS, EPS, MICRO_EPS, MODELS_ENUM,
    TYPO_ADD, TYPO_DEL, TYPO_NOCHANGE)
from utils.context_container import ContextContainer
from utils.edit_operations import detailed_edit_operations
from utils.logger import logger
from utils.utils import beam_search, take_first_n, makedirs

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


class Tuner:
    def __init__(self, config):
        self.config = config
        self.tuner_dir = config.tuner_dir
        if NUM_THREADS == 1:
            self.fixer = RNNBicontextTextFixer(self.config, from_tuner=True)
        else:
            self.fixer = None

    def get_data(self, pair):
        # import tensorflow as tf
        # tf.logging.set_verbosity(tf.logging.ERROR)
        if self.fixer is None:
            fixer = RNNBicontextTextFixer(self.config, from_tuner=True)
        else:
            fixer = self.fixer
        return fixer.get_actions_probabilities(pair)

    def train_data(self, gen, total=10000):

        weights_path = os.path.join(self.tuner_dir, 'final-weights.pkl')
        if os.path.isfile(weights_path):
            with open(weights_path, 'rb') as fl:
                res = pickle.load(fl)
                logger.log_info('loaded weights from', weights_path,
                                highlight=4)
                return res

        logger.start()
        iodata_path = os.path.join(self.tuner_dir, 'iodata.pkl')
        if os.path.isfile(iodata_path):
            with open(iodata_path, 'rb') as fl:
                X, Y = pickle.load(fl)
                logger.log_info('loaded io data from', iodata_path,
                                highlight=4)
        else:
            logger.log_info('constructing iodata', highlight=4)
            X = []
            Y = []
            sz = 0
            last = None
            gen = take_first_n(gen, total)
            # chunksize = (total + 4 * NUM_THREADS - 1) // (4 * NUM_THREADS)
            if NUM_THREADS > 1:
                with closing(multiprocessing.Pool(NUM_THREADS, maxtasksperchild=4)) as pool:
                    for i, (x, y) in tqdm(enumerate(pool.imap(
                            self.get_data, gen)), total=total):
                        X.append(x)
                        Y.append(y)
                        sz += len(x)
                        if last is None or sz > last + 10000:
                            logger.log_debug('%d/%d' % (i + 1, total), "with",
                                             sz, "examples so far..")
                            last = sz
                    pool.close()
                    pool.join()
            else:
                for i, (x, y) in tqdm(enumerate(map(
                        self.get_data, gen)), total=total):
                    X.append(x)
                    Y.append(y)
                    sz += len(x)
                    if last is None or sz > last + 1000:
                        logger.log_debug('%d/%d' % (i + 1, total), "with",
                                         sz, "examples so far..")
                        last = sz
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
            makedirs(iodata_path)
            with open(iodata_path, 'wb') as fl:
                pickle.dump((X, Y), fl)
                logger.log_info('dumping io data into', iodata_path,
                                highlight=4)
        import keras.backend as K
        from keras.layers import Input, Conv1D, Dense, Reshape
        from keras.models import Model
        from keras.optimizers import Adam, SGD, Adadelta
        from keras.losses import sparse_categorical_crossentropy
        from .custom_layers import Sparse

        logger.log_debug(X.shape, Y.shape)
        inp = Input((10,))
        combiner_val = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1]])
        sparse = Sparse()
        combiner = Dense(6, use_bias=False, activation='softmax')
        combiner.trainable = False
        model = Model(inp, combiner(sparse(inp)))
        model.summary()
        combiner.set_weights([combiner_val])
        lr = 0.1
        decay = 0.01
        model.compile(Adam(lr),
                      loss=sparse_categorical_crossentropy,
                      metrics=['sparse_categorical_accuracy'])
        bar = tqdm(range(2000), ncols=120)
        for epoch in bar:
            K.set_value(model.optimizer.lr, lr / (1 + epoch * decay))
            loss, acc = model.train_on_batch(X, Y)
            if epoch % 400 == 1:
                logger.log_debug('\n', sparse.get_weights()[0], '\n', sparse.get_weights()[1])
                logger.log_report("loss:", loss, "acc:", acc, highlight=2)
            bar.set_postfix(loss=loss, acc=acc)
            bar.refresh()

        loss, acc = model.evaluate(X, Y, verbose=0)
        logger.log_report('loss:', loss, 'acc:', acc, highlight=2)

        weights, bias = sparse.get_weights()
        makedirs(weights_path)
        with open(weights_path, 'wb') as fl:
            pickle.dump((weights, bias), fl)
            logger.log_info("logged weights into", weights_path, highlight=4)
        logger.log_full_report_into_file(weights_path)

        return weights, bias


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
        self.weights = [
            1.1919466, 0.45253742, 0.3623994, 0.82317746, -0.12727399,
            1.1804715, 1.1052611, 0.930409, 1.8024925, 1.8668712]
        self.bias = [
            4.154702, -5.471141, 1.381059, 1.3617834, 2.301759,
            2.3024273, 1.7338722, 1.7320414, 0.36447698, 0.36808118]
        self.use_look_forward = config.use_look_forward
        self.fixer_repr = config.fixer_repr

        if not from_tuner:
            logger.log_debug(self, ', beam:', self.beam_size,
                             ', delimiters only:', self.fix_delimiters_only,
                             ', look forward:', self.use_look_forward,
                             ', use default weights (no tuner):',
                             self.use_default_weights)
        self.weights = [
            1.1919466, 0.45253742, 0.3623994, 0.82317746, -0.12727399,
            1.1804715, 1.1052611, 0.930409, 1.8024925, 1.8668712]
        self.bias = [
            4.154702, -5.471141, 1.381059, 1.3617834, 2.301759,
            2.3024273, 1.7338722, 1.7320414, 0.36447698, 0.36808118]
        if self.use_default_weights:
            if not from_tuner:
                logger.log_info('USING default weights..', self.weights, self.bias)
        else:
            try:
                with open(os.path.join(config.tuner_dir, 'final-weights.pkl'), 'rb') as fl:
                    self.weights, self.bias = pickle.load(fl)
                if not from_tuner:
                    logger.log_info('loaded weights..\n', self.weights, '\n', self.bias)
            except Exception as err:
                if not from_tuner:
                    logger.log_error(err)
                    logger.log_error('weights not found, please train tuner!.. '
                                     'using default weights')
                    logger.log_info('loaded temp weights..\n', self.weights, '\n', self.bias)

    def __repr__(self):
        """
        Short description of the model and the architecture used.

        :rtype: str
        :returns: short description of the model
        """
        return self.fixer_repr

    def predict_occurence(self, str_before, str_after, dense_state_before,
                          dense_state_after, debug_tag=''):
        res_forward = np.log(self.forward_language_model.predict_likelihood(
            str_after, dense_state_before) + MICRO_EPS)
        res_backward = np.log(self.backward_language_model.predict_likelihood(
            str_before, dense_state_after) + MICRO_EPS)
        if debug_tag:
            self.debug_occurence(str_before, str_after, dense_state_before,
                                 dense_state_after, debug_tag,
                                 res_forward, res_backward)
        return res_forward + res_backward

    def debug_occurence(self, str_before, str_after, dense_state_before,
                        dense_state_after, debug_tag,
                        res_forward, res_backward):
        if DEBUG_MODE:
            backward_next = self.backward_language_model.debugger.get(
                dense_state_after[1:])
            forward_next = self.forward_language_model.debugger.get(
                dense_state_before[1:])
            logger.log_debug(
                "\n%s\n'%s'$$'%s'\n'%s'\t%.8f\n'%s'\t%.8f\n%.8f" % (
                    debug_tag, str_before, str_after,
                    backward_next, res_backward,
                    forward_next, res_forward,
                    res_backward + res_forward
                ))
            logger.log_seperator()

    def get_actions_probabilities(self, pair):
        correct_text, corrupt_text = pair
        all_probs = []
        all_labels = []
        # _t = datetime.now()
        # logger.log_debug(len(correct_text), len(corrupt_text))
        corrupt_states = self.backward_language_model.predict_seq(corrupt_text)
        # logger.log_debug('backward prediction', (datetime.now() - _t).total_seconds())
        # _t = datetime.now()
        correct_states = self.forward_language_model.predict_seq(correct_text)
        # logger.log_debug('forward prediction', (datetime.now() - _t).total_seconds())
        # _t = datetime.now()

        editops = detailed_edit_operations(corrupt_text, correct_text)
        # logger.log_debug('edits prediction', (datetime.now() - _t).total_seconds())
        # _t = datetime.now()
        for idx_corrupt, idx_correct, correct_operation in editops:  # tqdm(editops):
            if DEBUG_MODE:
                logger.log_seperator()
                logger.log_debug(
                    "\n" + "\033[31m" + correct_text[:idx_correct] +
                    '|\033[0m' + corrupt_text[idx_corrupt:] + "\n" +
                    str(correct_operation), highlight=4)
            probs = np.zeros((10,))
            X = correct_text[max(0, idx_correct - self.history_length): idx_correct]
            Y = corrupt_text[idx_corrupt + 1: idx_corrupt + self.history_length + 1]
            v = corrupt_text[idx_corrupt: idx_corrupt + 1]
            Xv = X + v
            Z = v + Y

            X_state = correct_states[idx_correct]
            Y_state = corrupt_states[idx_corrupt + 1]
            Z_state = corrupt_states[idx_corrupt]
            Xv_state = self.forward_language_model.predict(
                v, dense_state=X_state, return_dense_state=True)

            pr_del = 0.
            # TODO: This condition should be removed if it's needed for typo/non-typo fixing
            if v in self.tokenization_delimiters or not self.fix_delimiters_only:
                pr_del = self.predict_occurence(X, Y, X_state, Y_state, 'DEL')
            if v in self.tokenization_delimiters:
                probs[0] = pr_del
            else:
                probs[1] = pr_del
            # ###############################################################
            pr_nochg1 = self.predict_occurence(Xv, Y, Xv_state, Y_state, 'NOP1')
            pr_nochg2 = self.predict_occurence(X, Z, X_state, Z_state, 'NOP2')
            if v in self.tokenization_delimiters:
                probs[2] = pr_nochg1
                probs[3] = pr_nochg2
            else:
                probs[4] = pr_nochg1
                probs[5] = pr_nochg2

            # ###############################################################
            # try_to_addF = self.forward_language_model.predict_top_n(
            #     X.get_context())
            # try_to_addB = self.backward_language_model.predict_top_n(
            #     Z.get_context())
            # try_to_add = set(try_to_addF) | set(try_to_addB) | {' '}
            try_to_add = {' '}
            pr_add1_nosp, to_add1 = 0, None
            pr_add2_nosp, to_add2 = 0, None
            pr_add1_sp = 0
            pr_add2_sp = 0
            for s in try_to_add:
                if s == v == ' ':
                    continue
                Xs = X + s
                sZ = s + Z

                Xs_state = self.forward_language_model.predict(
                    s, dense_state=X_state, return_dense_state=True)
                sZ_state = self.backward_language_model.predict(
                    s, dense_state=Z_state, return_dense_state=True)

                pr_add1 = self.predict_occurence(X, sZ, X_state, sZ_state, 'ADD1 ' + s)
                pr_add2 = self.predict_occurence(Xs, Z, Xs_state, Z_state, 'ADD2 ' + s)
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
            if correct_operation == TYPO_DEL and v in self.tokenization_delimiters:
                # delete delimiter
                label = 0
            elif correct_operation == TYPO_DEL and v not in self.tokenization_delimiters:
                # delete char
                label = 1
            elif correct_operation == TYPO_NOCHANGE and v in self.tokenization_delimiters:
                # keep delimiter
                label = 2
            elif correct_operation == TYPO_NOCHANGE and v not in self.tokenization_delimiters:
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

            all_probs.append(probs)
            all_labels.append([label])
            # return probs + MICRO_EPS, label
        # logger.log_debug('XY predictions', (datetime.now() - _t).total_seconds())
        return np.array(all_probs), np.array(all_labels)

    def fix_step(self, text, future_states, score, before_context, dense_state,
                 cur_pos, res_text, last_op, added=0):
        """
        Fix a given state of the text.

        :param str text: The given corrupt text
        :param
        :param float score: Overall fixing score so far
        :param str before_context: The context before the character to be fixed
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
        if cur_pos >= len(text):
            return

        before_context_copy = before_context.copy()

        v = text[cur_pos]
        X = before_context.get_context()
        Xv = X + v
        Y = text[cur_pos + 1: cur_pos + self.history_length + 1]
        Z = text[cur_pos: cur_pos + self.history_length + 1]
        if DEBUG_MODE:
            logger.log_seperator()
            logger.log_debug(score, last_op, added, highlight=2)
            logger.log_debug("\n", before_context.get_context() + '|' +
                             text[cur_pos:], highlight=4)

        # ###############################################################
        if not self.fix_delimiters_only or v in self.tokenization_delimiters:
            pr_del = self.predict_occurence(X, Y, dense_state,
                                            future_states[cur_pos + 1], 'DEL')
            if v in self.tokenization_delimiters:
                pr_del = pr_del * self.weights[0] + self.bias[0]
            else:
                pr_del = pr_del * self.weights[1] + self.bias[1]
            yield (score - pr_del, before_context_copy.copy(), dense_state,
                   cur_pos + 1, res_text, TYPO_DEL, 0)

        # ###############################################################

        before_context = before_context_copy.copy()
        before_context.append_context(v)

        dense_state_new = self.forward_language_model.predict(
            v, dense_state=dense_state, return_dense_state=True)
        pr_nochg1 = self.predict_occurence(Xv, Y, dense_state_new,
                                           future_states[cur_pos + 1], 'NOP1')
        pr_nochg2 = self.predict_occurence(X, Z, dense_state,
                                           future_states[cur_pos], 'NOP2')
        if v in self.tokenization_delimiters:
            pr_nochg1 = pr_nochg1 * self.weights[2] + self.bias[2]
            pr_nochg2 = pr_nochg2 * self.weights[3] + self.bias[3]
        else:
            pr_nochg1 = pr_nochg1 * self.weights[4] + self.bias[4]
            pr_nochg2 = pr_nochg2 * self.weights[5] + self.bias[5]
        pr_nochg = pr_nochg1 + pr_nochg2

        yield (score - pr_nochg, before_context,
               dense_state_new, cur_pos + 1, res_text + v, TYPO_NOCHANGE, 0)

        # ###############################################################

        if added > 0:
            return
        try_to_add = {' '}
        if not self.fix_delimiters_only:
            # TODO: Add more candidates using dense_state
            pass
            # try_to_addF = self.forward_language_model.predict_top_n(
            #     X.get_context())
            # try_to_addB = self.backward_language_model.predict_top_n(
            #     Z.get_context())
            # try_to_add = set(try_to_addF) | set(try_to_addB) | {' '}

        for s in try_to_add:
            if v == s == ' ':
                continue
            pr_add_forward = np.log(MICRO_EPS)
            if self.use_look_forward:
                for shift in range(1, 3):
                    if cur_pos + shift >= len(future_states):
                        continue
                    cs = Z[:shift] + s
                    rsZ = cs + Z[shift:]

                    future_state_new = self.backward_language_model.predict(
                        cs, dense_state=future_states[cur_pos + shift],
                        return_dense_state=True)

                    pr_add_forward = max(
                        pr_add_forward,
                        self.predict_occurence(X, rsZ, dense_state, future_state_new))
                    # if pr_add_forward > pr_add: break

            Xs = X + s
            sZ = s + Z

            future_state_new = self.backward_language_model.predict(
                s, dense_state=future_states[cur_pos], return_dense_state=True)
            dense_state_new = self.forward_language_model.predict(
                s, dense_state=dense_state, return_dense_state=True)

            pr_add1 = self.predict_occurence(X, sZ, dense_state, future_state_new, 'ADD1 ' + s)
            pr_add2 = self.predict_occurence(Xs, Z, dense_state_new, future_states[cur_pos], 'ADD2' + s)
            if s in self.tokenization_delimiters:
                pr_add1 = pr_add1 * self.weights[6] + self.bias[6]
                pr_add2 = pr_add2 * self.weights[7] + self.bias[7]
            else:
                pr_add1 = pr_add1 * self.weights[8] + self.bias[8]
                pr_add2 = pr_add2 * self.weights[9] + self.bias[9]
            pr_add = pr_add1 + pr_add2

            if self.use_look_forward and (pr_add < pr_add_forward or pr_add < -30):
                pr_add = np.log(MICRO_EPS)

            before_context = before_context_copy.copy()
            before_context.append_context(s)
            yield (score - pr_add, before_context, dense_state_new, cur_pos,
                   res_text + s, (TYPO_ADD, s), added + 1)
        return

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
        future_states = self.backward_language_model.predict_seq(text)

        def is_terminal_state(state):
            """
            Local function to determine if a given state is terminal,
            in order to be used by beam search.

            :param tuple state: A state of the beam search
            :rtype: boolean
            :returns: boolean if the given state is terminal
            """
            (score, bef_context, dense_state, pos, res_text, last_op, added) = state
            return pos >= len(text)

        def next_states(state):
            """
            Local function to enumerate all the next states of a given state.

            :param tuple state: A state of the beam search
            :rtype: generator(state)
            :returns: an enumeration of all successor states of the given state
            """
            for next_state in self.fix_step(text, future_states, *state):
                yield next_state
            return

        before_context = ContextContainer(self.history_length)
        dense_state = self.forward_language_model.predict(return_dense_state=True)
        fixation = beam_search((0.0, before_context, dense_state, 0, '', 'beg', 0),
                               is_terminal_state, next_states,
                               # Compare the functions by costs: state[0]
                               (lambda x: x[0]), self.beam_size)
        logger.log_full_debug('total', len(text))
        logger.log_full_debug(text)
        logger.log_full_debug(fixation)
        score, _bef, _state, _pos, fixed_text, last_op, _ = fixation
        return fixed_text
