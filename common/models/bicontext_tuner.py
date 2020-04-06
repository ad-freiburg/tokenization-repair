"""
Module containing tuner of bicontext RNN fixers.


Copyright 2017-2018, University of Freiburg.

Mostafa M. Mohamed <mostafa.amin93@gmail.com>
"""
import gc
import multiprocessing
import os
import time
import warnings

from constants import (
    ACTIONS_DIM, LABELS_DIM, MICRO_EPS, NUM_THREADS,
    TYPO_ADD, TYPO_DEL, TYPO_NOCHANGE)
from utils.context_container import ContextContainer
from utils.logger import logger
from utils.edit_operations import detailed_edit_operations
from utils.utils import (
    cleanstr,
    gen_chunker, one_hot, open_or_create_write_file, makedirs, take_first_n)

import _pickle as pickle
import numpy as np


class BicontextTuner:
    """
    Decisions tuner that tunes the combination weights of the RNN Bicontext
    fixer. The input is a set of paths, each containing list of edit
    alignments <i, j, op> where i, j aligns character i from the corrupt text
    with j from the correct text with 'op' as a fixing operation.
    This input is transformed into a matrix X of size <?, 10>, denoting
    10 probabilities computed by the fixers, the probabilities of all
    actions at a given point, and a vector y of size <?, 6> of one-hot
    vectors with 1s corresponding to action op.
    The optimization task is defined by making this softmax{{Xw + b}M}
    as close as possible to y, by categorical cross entropy.
    """
    def __init__(self, config):
        self.fixer_config = config.copy()
        self.fixer_config.use_default_weights = False
        self.save_dir = config.tuner_save_dir
        self.load_dir = config.tuner_load_dir
        makedirs(self.save_dir)

    def load_weights(self):
        """
        Load the weights from the saved pickle file.

        :returns: Pair of vectors, the weights and biases respectively.
        :rtype: pair(ndarray, ndarray)
        """
        path = os.path.join(self.load_dir, 'final-weights.p')
        with open(path, 'rb') as fil:
            content = pickle.load(fil)
            assert len(content) >= 2
            weights = content[0]
            bias = content[1]
        assert weights.shape == (10, 1) or weights.shape == (10,)
        assert bias.shape == (10, 1) or bias.shape == (10,)
        logger.log_info('loaded weights..', path, highlight=2)
        return weights, bias

    def get_fixing_actions_probabilities_(self, tpath):
        """
        Construct the input data corresponding to a text and its corrupt
        version.

        :param pair tpath: Pair of file index and edit file path string
        :rtype: pair(ndarray, ndarray)
        :returns:
            Corresponding input probabilities matrix and
            output decisions vector.
        """
        t, path = tpath
        correct_text, corrupt_text, edit_ops = path
        return self.get_fixing_actions_probabilities_helper(
            correct_text, corrupt_text, edit_ops)

    def get_fixing_actions_probabilities(self, pair):
        correct_text, corrupt_text = pair
        #logger.log_debug(correct_text[:30], '---', corrupt_text[:30])
        #return
        correct_text = cleanstr(correct_text)
        corrupt_text = cleanstr(corrupt_text)
        edit_ops = detailed_edit_operations(corrupt_text, correct_text)
        return self.get_fixing_actions_probabilities_helper(
            correct_text, corrupt_text, edit_ops)

    def get_fixing_actions_probabilities_helper(self, correct_text,
                                                corrupt_text, edit_ops):
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        from .rnn_bicontext_fixer import RNNBicontextTextFixer
        fixer = RNNBicontextTextFixer(self.fixer_config, from_tuner=True)
        actions_matrix = np.zeros((len(edit_ops), ACTIONS_DIM))
        actions = np.zeros((len(edit_ops),))
        tot = 0
        tic = time.time()
        num_nop, num_del, num_add = 0, 0, 0
        for edit_op in edit_ops:
            idx_corrupt, idx_correct, op = edit_op

            if idx_corrupt >= len(corrupt_text):
                continue
            before_context = ContextContainer(fixer.history_length)
            after_context = ContextContainer(fixer.history_length)
            before_context.append_context(correct_text[:idx_correct])
            after_context.push_context(corrupt_text[idx_corrupt + 1:])
            current = corrupt_text[idx_corrupt]
            assert (isinstance(op, tuple) and op[0] == TYPO_ADD) or op in\
                [TYPO_NOCHANGE, TYPO_DEL], op
            if op == TYPO_NOCHANGE:
                num_nop += 1
            elif op == TYPO_DEL:
                num_del += 1
            else:
                num_add += 1
            res = fixer.get_actions_probabilities(
                before_context, after_context, current, op)
            if res is not None:
                X, y = res
                actions_matrix[tot] = X
                actions[tot] = y
                tot += 1
        toc = time.time()
        logger.log_debug('fix ops:', int(toc - tic), 'sec',  # t,
                         len(edit_ops), len(correct_text),
                         num_nop, num_del, num_add,
                         correct_text[:100], '----', corrupt_text[:100],
                         highlight=1)
        # del correct_text, corrupt_text, edit_ops, fixer; gc.collect()
        actions_matrix = actions_matrix[:tot, :]
        actions = actions[:tot].reshape(-1, 1)
        return actions_matrix, actions

    def _preprocess_data(self, paths):
        """
        Preprocess the input data from a set of edit files and the
        corresponding input data.

        :param list[str] paths:
            The path of the input file, pickled file consisisting of
            list of triples (i, j, op), where op is TYPO_DEL, TYPO_NOCHG,
            or (TYPO_ADD, chr).
        :rtype: pair(ndarray, ndarray)
        :returns:
            Corresponding input probabilities matrix and
            output decisions vector.
        """
        io_pairs = []
        chunksize = (len(paths) + 4 * NUM_THREADS - 1) // (4 * NUM_THREADS)
        logger.log_info('starting preprocessing..', highlight=1)
        assert chunksize <= len(paths) and chunksize > 0
        with multiprocessing.Pool(NUM_THREADS) as pool:
            for done, io_pair in enumerate(pool.imap_unordered(
                    self.get_fixing_actions_probabilities,
                    enumerate(paths), chunksize=chunksize)):
                io_pairs.append(io_pair)
                if (done + 1) % 1 == 0:
                    logger.log_info('%d/%d done..' % (done + 1, len(paths)),
                                    highlight=1)
                    #logger.log_full_report_into_file(self.save_dir, keep_log=True)
            pool.close()
            pool.join()
        logger.log_info('done preprocessing..', highlight=1)
        #logger.log_full_report_into_file(self.save_dir, keep_log=True)
        inp, out = zip(*io_pairs)
        del io_pairs
        gc.collect()
        X = np.vstack(inp) + MICRO_EPS
        y = np.vstack(out).squeeze()
        logger.log_debug(X.shape, y.shape)
        yonehot = one_hot(y, LABELS_DIM)
        return X, yonehot

    def preprocess_data(self, pairs_gen):
        io_pairs = []
        chunksize = 5
        total = 2100
        logger.log_info('starting preprocessing..', highlight=1)
        gen = gen_chunker(take_first_n(pairs_gen, total),
                          NUM_THREADS * chunksize)
        logger.log_info('made generator..', highlight=1)

        total_done = 0
        acc_sz = 0
        for batch in gen:
            with multiprocessing.Pool(NUM_THREADS) as pool:
                for done, io_pair in enumerate(pool.imap(
                        self.get_fixing_actions_probabilities,
                        batch, chunksize=chunksize
                        )):
                    io_pairs.append(io_pair)
                    total_done += 1
                    acc_sz += np.shape(io_pair[0])[0]
                    logger.log_debug(np.shape(io_pair[0]), np.shape(io_pair[1]))
                    gc.collect()
                    if (total_done + 1) % 10 == 0:
                        logger.log_info('%d/%d done.. %d' % (
                            total_done + 1, total, acc_sz), highlight=1)
                        logger.log_full_report_into_file(self.save_dir, keep_log=True)
                gc.collect()
                pool.close()
                pool.join()
            gc.collect()
        logger.log_info('done preprocessing..', highlight=1)
        logger.log_full_report_into_file(self.save_dir, keep_log=True)
        X, y = zip(*io_pairs)
        del io_pairs
        gc.collect()
        #for x in X: logger.log_debug(np.shape(x))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        logger.log_debug(np.shape(X), np.shape(y))
        yonehot = one_hot(y, LABELS_DIM)
        return X, yonehot

    def construct_variables(self, dim):
        """
        Construct the tensorflow variables.

        :param int dim: The dimension of the weights.
        """
        import tensorflow as tf
        self.weights = tf.Variable(tf.random_normal([dim], seed=41) * 0.01 + 1)
        self.bias = tf.Variable(tf.random_normal([dim], seed=41) * 0.01)
        self.global_step = tf.Variable(1, name='global_step', trainable=False)
        self.variables = [self.global_step, self.weights, self.bias]

    def construct_weights(self, actions_matrix_shape, actions_shape):
        """
        Construct the tensors to train the tuner.

        :param pair actions_matrix_shape:
            The input matrix shape, this should be (None, 10)
        :param pair actions_shape:
            The shape of the actions matrix, this should be (None, 6)
        """
        import tensorflow as tf
        self.tfX = tf.placeholder(tf.float32, shape=actions_matrix_shape,
                                  name='input')
        self.tfy = tf.placeholder(tf.float32, shape=actions_shape,
                                  name='output')
        self.construct_variables(actions_matrix_shape[1])
        combiner = tf.constant([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 1]],
                               dtype=tf.float32)
        assert combiner.shape == (ACTIONS_DIM, LABELS_DIM)
        self.predicted_actions = tf.matmul(
            self.weights * tf.log(self.tfX) + self.bias,
            combiner)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.tfy, logits=self.predicted_actions))

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_step)

    def train(self, paths, epochs=10000):
        """
        Train the tuner weights using the edit files given as arguments.

        :param args(str) paths:
            The paths of the input files, pickled files consisisting of
            list of triples <i, j, op>, where op is TYPO_DEL, TYPO_NOCHG,
            or <TYPO_ADD, chr>.
        :param int epochs: The number of training epochs
        """
        weights_path = os.path.join(self.load_dir, 'final-weights.p')
        if os.path.isfile(weights_path):
            logger.log_info('%s weights found..' % weights_path)
            return None

        logger.start()
        load_dat_file = os.path.join(self.load_dir, 'iodata.p')
        save_dat_file = os.path.join(self.save_dir, 'iodata.p')
        if os.path.isfile(load_dat_file):
            X, y = pickle.load(open(load_dat_file, 'rb'))
            logger.log_info('loaded data from', load_dat_file)
        else:
            logger.log_info('%s not found, tuner will be trained..' % load_dat_file)
            X, y = self.preprocess_data(paths)
            with open_or_create_write_file(save_dat_file, 'wb') as fl:
                pickle.dump((X, y), fl)
                logger.log_info('dumped data into', save_dat_file)

        p = multiprocessing.Process(target=self._train, args=(X, y, epochs))
        p.start()
        p.join()
        logger.log_report('training done..')
        logger.log_full_report_into_file(self.save_dir)

    def _train(self, X, y, epochs):
        """
        Train the tuner given the training data.

        :param ndarray X: ? x 10 matrix of the input probabilities
        :param ndarray y: ? x 1 vector of the output labels
        :param int epochs: Number of training epochs.
        :rtype: pair(ndarray)
        :returns: Pair of the weights and biases vectors.
        """
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        warnings.filterwarnings("ignore")
        self.construct_weights((None,) + X.shape[1:], (None,) + y.shape[1:])
        with tf.Session() as sess:
            losses = []
            sess.run(tf.global_variables_initializer())
            for epoch in range(1, 1 + epochs):
                learning_rate = 0.01
                _, loss_val = sess.run(
                    [self.optimizer, self.loss],
                    feed_dict={self.tfX: X, self.tfy: y,
                               self.learning_rate: learning_rate})
                losses.append(loss_val)
                logger.log_info('loss %d: %.6f' % (epoch, loss_val))

                if epoch % 200 == 0:
                    _, weights, bias = sess.run(
                        self.variables, feed_dict={self.tfX: X, self.tfy: y})
                    self.save(weights, bias, losses, epoch)
            _, weights, bias = sess.run(self.variables)
            self.save(weights, bias, losses, epochs)
            self.save(weights, bias, losses)

    def save(self, weights, bias, losses, it=None):
        if it is None:
            dump_path = os.path.join(self.save_dir, 'final-weights.p')
        else:
            dump_path = os.path.join(self.save_dir, 'it%.6d-final-weights.p' % it)
        with open_or_create_write_file(dump_path, 'wb') as fl:
            pickle.dump((weights, bias, losses), fl)
        if it is not None:
            logger.log_info('dumped model.. %d epochs' % it, dump_path)
        else:
            logger.log_info('dumped model..', dump_path)
        logger.log_info('weights:', weights.tolist())
        logger.log_info('biases:', bias.tolist())
