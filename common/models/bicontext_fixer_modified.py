"""
Module containing bicontext text fixer using two directional RNN
character-based model.
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
        self.use_kernel = config.use_kernel
        self.use_bias = config.use_bias
        self.fix_delimiters_only = config.fix_delimiters_only
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
                        if last is None or sz > last + 50000:
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
                    if last is None or sz > last + 50000:
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
        assert self.fix_delimiters_only, "The possibilities are not handled for non delimiters"
        return self.fit(X, Y)

    def create_network(self, X, Y, combiner_val):
        bias = np.zeros(X.shape[1:])
        weights = np.ones(X.shape[1:])
        if X.shape[0] == 0:
            return weights, bias

        import keras.backend as K
        from keras.layers import Input, Conv1D, Dense, Reshape, Activation, Lambda, Add
        from keras.models import Model
        from keras.optimizers import Adam, SGD, Adadelta
        from .custom_layers import Sparse

        logger.log_debug("\n", X.shape, Y.shape,
                         np.unique(Y, return_counts=True), '\n',
                         X.mean(axis=0), '\n\n', X.std(axis=0))
        inp = Input(X.shape[1:])
        sparse = Sparse(use_kernel=self.use_kernel, use_bias=self.use_bias)
        if X.ndim > 2:
            sum_layer = Lambda(lambda x: K.sum(x, axis=-1))
        else:
            sum_layer = Activation('linear')
        print(combiner_val.shape)
        combiner = Dense(combiner_val.shape[-1], activation='softmax', use_bias=False)
        combiner.trainable = False
        out = combiner(sum_layer(sparse(inp)))
        combiner.set_weights([combiner_val])
        model = Model(inp, out)
        model.summary()

        lr = 0.1
        decay = 5e-2
        model.compile(
            Adam(lr),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])
        bar = tqdm(range(10000), ncols=160)
        for epoch in bar:
            K.set_value(model.optimizer.lr, lr / (1 + epoch * decay))
            vals = model.train_on_batch(X, Y)
            # vals = model.fit(X, Y, verbose=0, batch_size=min(X.shape[0], 1 << 17))
            names = [name.replace('sparse_categorical_accuracy', 'acc')
                    for name in model.metrics_names]
            dicts = dict(zip(names, vals))
            if epoch % 400 == 1:
                for arr in sparse.get_weights():
                    logger.log_debug("\n" + str(np.round(arr, 3)))
            bar.set_postfix(**dicts)
            bar.refresh()

        loss, acc = model.evaluate(X, Y, verbose=0)
        logger.log_report('loss:', loss, 'acc:', acc, highlight=2)

        if self.use_kernel:
            weights = sparse.get_weights()[0]
        if self.use_bias:
            bias = sparse.get_weights()[-1]
        return weights, bias

    def fit(self, X, Y):

        weights_path = os.path.join(self.tuner_dir, 'final-weights.pkl')
        logger.log_debug(X.shape, Y.shape, np.unique(Y, return_counts=True))
        #logger.log_debug("\n", np.round(X.mean(axis=0), 5))
        #logger.log_debug("\n", np.round(X.std(axis=0), 5))
        msk_dels = (Y[:, 0] == 0) | (Y[:, 0] == 1)
        msk_adds = (Y[:, 0] == 2) | (Y[:, 0] == 3)

        X_adds = X[msk_adds]
        X_adds = X_adds[:, np.array([3, 4, 5]), ...]
        Y_adds = Y[msk_adds]
        Y_adds[Y_adds == 2] = 1
        Y_adds[Y_adds == 3] = 0
        combiner_adds = np.array(
            [[1, 0],
             [1, 0],
             [0, 1]])

        X_dels = X[msk_dels]
        X_dels = X_dels[:, np.array([0, 1, 2]), ...]
        Y_dels = Y[msk_dels]
        Y_dels[Y_dels == 0] = 1
        Y_dels[Y_dels == 1] = 0
        combiner_dels = np.array(
            [[1, 0],
             [1, 0],
             [0, 1]])

        kernel_dels, bias_dels = self.create_network(X_dels, Y_dels, combiner_dels)
        kernel_adds, bias_adds = self.create_network(X_adds, Y_adds, combiner_adds)

        weights = np.ones(X.shape[1:])
        bias = np.zeros(X.shape[1:])
        weights[np.array([0, 1, 2])] = kernel_dels
        bias[np.array([0, 1, 2])] = bias_dels

        weights[np.array([3, 4, 5])] = kernel_adds
        bias[np.array([3, 4, 5])] = bias_adds
        logger.log_debug("\n", np.round(weights, 3), "\n", np.round(bias, 3))

        #weights, bias = sparse.get_weights()
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

        self.bidirectional_weights = config.bidirectional_weights
        self.bidir = config.bidir
        self.lflen = config.lflen
        self.beam_size = config.beam_size
        self.history_length = config.history_length
        self.alphabet = DECODER_DICT
        self.tokenization_delimiters = ' \t'
        self.fix_delimiters_only = config.fix_delimiters_only
        if self.bidirectional_weights:
            self.weights = np.ones((6, 2))
            self.bias = np.zeros((6, 2))
        else:
            self.weights = np.ones((6,))
            self.bias = np.zeros((6,))
        self.fixer_repr = config.fixer_repr

        if not from_tuner:
            logger.log_debug(self, ', beam:', self.beam_size,
                             ', delimiters only:', self.fix_delimiters_only)
        try:
            with open(os.path.join(config.tuner_dir, 'final-weights.pkl'), 'rb') as fl:
                self.weights, self.bias = pickle.load(fl)
            if not from_tuner:
                logger.log_info('loaded weights..\n', np.round(self.weights,
                    3), '\n', np.round(self.bias, 3))
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
        context_lf = self.lflen
        res_forward = np.log(self.forward_language_model.predict_likelihood(
            str_after, dense_state_before, small_context_size=context_lf) + MICRO_EPS)
        if self.bidir:
            res_backward = np.log(self.backward_language_model.predict_likelihood(
                str_before, dense_state_after, small_context_size=context_lf) + MICRO_EPS)
        else:
            res_backward = 0
        if debug_tag:
            self.debug_occurence(str_before, str_after, dense_state_before,
                                 dense_state_after, debug_tag,
                                 res_forward, res_backward)
        if self.bidirectional_weights:
            return np.array([res_forward, res_backward])
        else:
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
        corrupt_states = self.backward_language_model.predict_seq(corrupt_text)
        correct_states = self.forward_language_model.predict_seq(correct_text)

        editops = detailed_edit_operations(corrupt_text, correct_text)
        for idx_corrupt, idx_correct, correct_operation in editops:  # tqdm(editops):
            if idx_correct > 0 and correct_text[idx_correct - 1] in self.tokenization_delimiters:
                continue
            if DEBUG_MODE:
                logger.log_seperator()
                logger.log_debug(
                    "\n" + "\033[31m" + correct_text[:idx_correct] +
                    '|\033[0m' + corrupt_text[idx_corrupt:] + "\n" +
                    str(correct_operation), highlight=4)
            if self.bidirectional_weights:
                probs = np.log(np.zeros((6, 2)) + MICRO_EPS)
            else:
                probs = np.log(np.zeros((6,)) + MICRO_EPS)

            X = correct_text[max(0, idx_correct - self.history_length): idx_correct]
            Y = corrupt_text[idx_corrupt + 1: idx_corrupt + self.history_length + 1]
            v = corrupt_text[idx_corrupt: idx_corrupt + 1]
            s = ' '
            Xv = X + v
            Xvs = Xv + s
            sY = s + Y
            vY = v + Y

            X_state = correct_states[idx_correct]
            Y_state = corrupt_states[idx_corrupt + 1]
            vY_state = corrupt_states[idx_corrupt]
            Xv_state = self.forward_language_model.predict(
                v, dense_state=X_state, return_dense_state=True)
            Xvs_state = self.forward_language_model.predict(
                s, dense_state=Xv_state, return_dense_state=True)
            sY_state = None
            if self.bidir:
                sY_state = self.backward_language_model.predict(
                    s, dense_state=Y_state, return_dense_state=True)

            if v in self.tokenization_delimiters:
                probs[0] = self.predict_occurence(Xv, Y, Xv_state, Y_state, 'NOP1')
                probs[1] = self.predict_occurence(X, vY, X_state, vY_state, 'NOP2')
                probs[2] = self.predict_occurence(X, Y, X_state, Y_state, 'DEL')
            else:
                probs[3] = self.predict_occurence(Xvs, Y, Xvs_state, Y_state, 'ADD1')
                probs[4] = self.predict_occurence(Xv, sY, Xv_state, sY_state, 'ADD2')
                probs[5] = self.predict_occurence(Xv, Y, Xv_state, Y_state, 'NOP')

            label = -1
            if correct_operation == TYPO_DEL and v in self.tokenization_delimiters:
                # delete delimiter
                label = 0
            elif correct_operation == TYPO_NOCHANGE and v in self.tokenization_delimiters:
                # keep delimiter
                label = 1
            elif correct_operation == TYPO_NOCHANGE and v not in self.tokenization_delimiters:
                # keep char
                label = 2
            elif (isinstance(correct_operation, tuple) and
                  correct_operation[0] == TYPO_ADD and
                  correct_operation[1] in self.tokenization_delimiters):
                # add delimiter
                label = 3
            else:
                assert False, 'invalid operation %s' % str(correct_operation)

            all_probs.append(probs)
            all_labels.append([label])
            # return probs + MICRO_EPS, label
        # logger.log_debug('XY predictions', (datetime.now() - _t).total_seconds())
        return np.array(all_probs), np.array(all_labels)

    def fix_step(self, text, future_states, score, before_context, dense_state,
                 cur_pos, res_text, last_op):
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
        assert text[cur_pos] != ' '

        if DEBUG_MODE:
            logger.log_seperator()
            logger.log_debug("\n", before_context.get_context() + '|||' +
                             text[cur_pos:], highlight=4)

        # next_idx
        next_idx = cur_pos + 1
        if next_idx < len(text) and text[next_idx] == ' ':
            next_idx = next_idx + 1
            assert text[next_idx] != ' '
        is_prev_space = False
        if cur_pos > 0 and text[cur_pos] == ' ':
            is_prev_space = True
        action = ''  # TODO: fill for debugging

        # needed states
        future_state = future_states[cur_pos]
        dense_state_space = self.forward_language_model.predict(
            ' ', dense_state=dense_state, return_dense_state=True)
        if self.bidir:
            future_state_space = self.backward_language_model.predict(
                ' ', dense_state=future_state, return_dense_state=True)
        else:
            future_state_space = None

        # contexts
        X = before_context.get_context()
        Xs = X + ' '
        Z = text[cur_pos: cur_pos + self.history_length + 1]
        sZ = ' ' + Z

        pr_sp1 = self.predict_occurence(Xs, Z, dense_state_space, future_state, 'sp')
        pr_sp2 = self.predict_occurence(X, sZ, dense_state, future_state_space, 'sp_nxt')
        pr_nosp = self.predict_occurence(X, Z, dense_state, future_state, 'nosp')

        if is_prev_space:
            pr_sp = np.sum(pr_sp1 * self.weights[0] + self.bias[0]) + np.sum(pr_sp2 * self.weights[1] + self.bias[1])
            pr_nosp = np.sum(pr_nosp * self.weights[2] + self.bias[2])
        else:
            pr_sp = np.sum(pr_sp1 * self.weights[3] + self.bias[3]) + np.sum(pr_sp2 * self.weights[4] + self.bias[4])
            pr_nosp = np.sum(pr_nosp * self.weights[5] + self.bias[5])

        before_context = before_context.copy()
        before_context_copy = before_context.copy()
        #######
        # there is no space
        dense_state_new = self.forward_language_model.predict(
            text[cur_pos], dense_state=dense_state, return_dense_state=True)
        before_context.append_context(text[cur_pos])
        yield (score - pr_nosp, before_context.copy(), dense_state_new,
               next_idx, res_text + text[cur_pos], action)

        # There is a space
        dense_state_new = self.forward_language_model.predict(
            text[cur_pos], dense_state=dense_state_space, return_dense_state=True)
        before_context_copy.append_context(' ')
        before_context_copy.append_context(text[cur_pos])
        yield (score - pr_sp, before_context_copy.copy(), dense_state_new,
               next_idx, res_text + ' ' + text[cur_pos], action)
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
            (score, bef_context, dense_state, pos, res_text, last_op) = state
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

        assert '  ' not in text
        before_context = ContextContainer(self.history_length)
        dense_state = self.forward_language_model.predict(return_dense_state=True)
        fixation = beam_search((0.0, before_context, dense_state, 0, '', 'beg'),
                               is_terminal_state, next_states,
                               # Compare the functions by costs: state[0]
                               (lambda x: x[0]), self.beam_size)
        logger.log_full_debug('total', len(text))
        logger.log_full_debug(text)
        logger.log_full_debug(fixation)
        score, _bef, _state, _pos, fixed_text, last_op = fixation
        return fixed_text
