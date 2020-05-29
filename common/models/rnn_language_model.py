import os
import gc
import numpy as np

from .character_model import CharacterModel
from constants import BACKWARD, FORWARD, NUM_THREADS, EOS, UNK, SOS
from utils.cacher import Cacher, Debugger, hashstate
from utils.logger import logger


DEBUG_MODE = False


class RNNLanguageModel(CharacterModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path

        self.history_length = config.history_length
        self.perturbate = config.perturbate
        self.return_sequences = config.return_sequences
        self.rnn_layers = config.rnn_layers
        self.rnn_type = config.rnn_type
        self.model_repr = config.model_repr
        self.rnn_units = config.rnn_units
        self.fully_connected_layers = config.fully_connected_layers
        self.fully_connected_units = config.fully_connected_units
        self.dropout_rate = config.dropout_rate
        self.learning_rate = config.learning_rate
        self.decay_rate = config.decay_rate
        self.batch_size = config.batch_size
        self.direction = config.direction
        self.inference = config.inference
        self.initial_epoch = 0
        self.epochs = config.epochs

        self.output_name = 'char_' + self.direction
        self.losses = {self.output_name: 'sparse_categorical_crossentropy'}
        from models.custom_layers import sparse_accuracy, sparse_top_k_categorical_acc
        self.metrics = {self.output_name: [sparse_accuracy, sparse_top_k_categorical_acc()]}

        self.debugger = Debugger()  # TODO: remove debugger
        self.cacher = Cacher(4000)

        """
        if self.inference:
            import tensorflow as tf
            with tf.device('/cpu:0'):
                self.load_or_create_model()
        else:
            self.load_or_create_model()
        """
        self.load_or_create_model()

    def load_or_create_model(self):
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        # if not self.load_model(self.model_load_path):
        from models.custom_layers import Onehot
        from keras.layers import (
            Activation, Input, LSTM, GRU, Dropout,
            Dense, TimeDistributed, Lambda)
        from keras.models import Model

        inp = Input((None,))
        rnn_layer = Onehot(self.alphabet_size)(inp)
        initial_states = [
            (Input((self.rnn_units,)), Input((self.rnn_units,)))
            for layer in range(self.rnn_layers)]
        output_states = []
        flat_initial_states = []
        for initial_state in initial_states:
            flat_initial_states.extend(initial_state)

        for i in range(self.rnn_layers):
            ret_seq = self.return_sequences or i < self.rnn_layers - 1
            go_backwards = (i == 0) and (self.direction == BACKWARD)
            rnn_cell = {'GRU': GRU, 'LSTM': LSTM}[self.rnn_type]
            rnn_cell = rnn_cell(
                self.rnn_units,
                #, initial_state=initial_states[i])
                recurrent_activation='sigmoid',
                return_sequences=ret_seq, return_state=True,
                go_backwards=go_backwards)
            rnn_layer, state_c, state_h = rnn_cell(
                rnn_layer, initial_state=initial_states[i])
            output_states.extend([state_c, state_h])
            # print(state_c.shape, state_h.shape)

        dense_layer = Dropout(self.dropout_rate)(rnn_layer)

        for _ in range(self.fully_connected_layers):
            if self.return_sequences:
                dense_layer = TimeDistributed(Dense(
                    self.fully_connected_units,
                    activation='relu'))(dense_layer)
            else:
                dense_layer = Dense(
                    self.fully_connected_units,
                    activation='relu')(dense_layer)

        if self.return_sequences:
            dense_layer = TimeDistributed(Dense(
                self.alphabet_size), name='logits')(dense_layer)
        else:
            dense_layer = Dense(
                self.alphabet_size, name='logits')(dense_layer)

        if self.direction == BACKWARD:
            dense_layer = Lambda(lambda x: x[:, ::-1, :], name='reverse')(
                dense_layer)
        output = Activation('softmax', name=self.output_name)(dense_layer)

        self.model = Model([inp] + flat_initial_states, outputs=[output] + output_states,
                           name='%s_LM' % self.direction)
        self.model.summary()

        if os.path.isfile(self.model_load_path):
            self.model.load_weights(self.model_load_path)
            rweights = rnn_cell.get_weights()
            rweights[-1][1024: 2048] += 1
            rnn_cell.set_weights(rweights)
            logger.log_info('loaded weights from', self.model_load_path, highlight=2)
        else:
            logger.log_error('weights not found..', self.model_load_path, highlight=1)
        try:
            self.compile()
        except Exception as err:
            logger.log_error("Couldn't compile model", err, highlight=1)

    def make_generator(self, gen):
        # TODO
        # TODO: also use new hidden state
        pass

    def train(self, train_data):
        lines, generator = train_data
        train_gen = self.make_generator(generator)
        return self.model.fit_generator(
            self.make_generator(generator),
            train_gen,
            steps_per_epoch=steps,
            verbose=1,
            workers=NUM_THREADS,
            use_multiprocessing=True,
            callbacks=[self.create_saver_callback(5, 2)])

    def __repr__(self):
        return self.model_repr

    def new_state(self, batch_size=1):
        if self.rnn_type == 'LSTM':
            return [np.zeros((batch_size, self.rnn_units))
                    for _ in range(self.rnn_layers * 2)]
        elif self.rnn_type == 'GRU':
            # TODO: verify
            return [np.zeros((batch_size, self.rnn_units))
                    for _ in range(self.rnn_layers)]

    def adjust_input(self, text, state, encode=True, include_pad=False):
        if encode:
            text_codes = self.str_codes(text)
        else:
            text_codes = text

        if len(text_codes) == 0 or include_pad:
            if self.direction == FORWARD:
                text_codes = [SOS] + text_codes.tolist()
            else:
                text_codes = text_codes.tolist() + [EOS]
        text_codes = np.array(text_codes, dtype=np.uint8)

        inp = [np.expand_dims(text_codes, axis=0)] + state
        return inp

    def predict(self, text='', state=None, dense_state=None, return_state=False,
                return_dense_state=False,
                encode=True, return_sequence=False, include_pad=False):
        assert state is None or dense_state is None, "Only one of the states can be used"
        assert not return_sequence or self.return_sequences, "Can't return sequence if the model doesn't"
        assert not return_state or not return_dense_state, "Can't return both dense and non-dense states"

        if state is None and dense_state is None:
            state = self.new_state()
        elif state is None:
            state = dense_state[1:]

        inp = self.adjust_input(text, state, encode=encode, include_pad=include_pad)
        outputs = self._predict(inp)
        probs, state = outputs[0][0], outputs[1:]
        if not return_sequence and self.return_sequences:
            if self.direction == BACKWARD:
                probs = probs[0]
            if self.direction == FORWARD:
                probs = probs[-1]

        if return_state:
            return probs, state
        elif return_dense_state:
            return [probs] + state
        else:
            return probs

    def predict_seq(self, text):
        predictions = [self.predict(return_dense_state=True)]
        if self.direction == BACKWARD:
            text = text[::-1]
        for c in text:
            predictions.append(self.predict(c, dense_state=predictions[-1],
                                            return_dense_state=True))
        if self.direction == BACKWARD:
            predictions = predictions[::-1]
        return predictions

    def _predict_likelihood(self, text_codes, dense_state):
        if self.direction == BACKWARD:
            text_codes = text_codes[1:]
            text_codes = text_codes[::-1]
        else:
            text_codes = text_codes[:-1]
        predictions = [dense_state]
        for c in text_codes:
            predictions.append(self.predict(np.array([c], dtype=text_codes.dtype),
                                            dense_state=predictions[-1],
                                            return_dense_state=True,
                                            encode=False))
        predictions = predictions[1:]
        predictions = [x[0] for x in predictions]
        return predictions

    def push_debugger(self, initial_state, inp, final_state):
        prev = self.debugger.get(initial_state)
        if self.direction == BACKWARD:
            res = self.codes_to_str(inp[0][0]) + prev
        elif self.direction == FORWARD:
            res = prev + self.codes_to_str(inp[0][0])
        self.debugger.insert(final_state, res)

    def _predict(self, inp):
        hinp = hashstate(inp)
        outputs = self.cacher.get_cached_value(hinp)
        if outputs is not None:
            return outputs
        outputs = self.model.predict(inp)
        self.cacher.add_cache_value(hinp, outputs)
        if DEBUG_MODE:
            self.push_debugger(inp[1:], inp, outputs[1:])
        return outputs

    def predict_likelihood(self, text, dense_state, small_context_size=2, encode=True):
        if encode:
            text_codes = self.str_codes(text)
        else:
            text_codes = text

        if self.direction == BACKWARD:
            text_codes = np.array([SOS] + text_codes.tolist(), dtype=text_codes.dtype)
            text_codes = text_codes[-small_context_size:]
        else:
            text_codes = np.array(text_codes.tolist() + [EOS], dtype=text_codes.dtype)
            text_codes = text_codes[:small_context_size]

        prev_pred, state = dense_state[0], dense_state[1:]
        # inp = [np.expand_dims(text_codes, axis=0)] + state
        # outputs = self._predict(inp)[0][0].tolist()
        # outputs_a = self._predict_likelihood(text_codes, dense_state)
        # assert np.allclose(outputs, outputs_a, rtol=1e-4, atol=1e-6), (np.shape(outputs), np.shape(outputs_a))
        outputs = self._predict_likelihood(text_codes, dense_state)

        if self.direction == BACKWARD:
            preds = [prev_pred] + outputs
            text_codes = text_codes[::-1]
        else:
            preds = [prev_pred] + outputs
        assert len(preds) == len(text_codes)

        ans = []
        for t in range(len(preds)):
            acc = 0
            for i in range(len(preds) - 1 - t, -1, -1):
                p = preds[i][text_codes[i]]
                acc = p * (acc + 1)
            ans.append(acc)  # / (len(preds) - t))
        # logger.log_debug(ans)
        return ans[0]

    def predict_top_n(self, text, n=1):
        probs = self.predict(text)
        if n > 1:
            return self.codes_to_str(probs.argsort()[-n:])
        else:
            return self.codes_to_str([probs.argmax()])

    def sample_analysis(self):
        for temp in [0.3, 0.5, 0.6, 1.2]:
            text = 'Hello America, this is the new capitalism'
            state = None
            char = self.str_codes(text)
            for _ in range(500):
                preds, state = self.predict(char, state=state, return_state=True, encode=False)
                # preds = self.predict(text)
                char = self.sample_char(preds, temperature=temp)
                if self.direction == BACKWARD:
                    text = char + text
                elif self.direction == FORWARD:
                    text = text + char
                char = self.str_codes(char)
            logger.log_report('Temperature %.2f\n' % temp, text)
            logger.log_seperator()
