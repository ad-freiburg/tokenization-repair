import gc
import numpy as np

from .character_model import CharacterModel
from constants import BACKWARD, FORWARD, NUM_THREADS, EOS, UNK, SOS
from utils.logger import logger
from utils.cacher import Cacher


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

        self.predictions_cache = Cacher(500)
        if self.inference:
            import tensorflow as tf
            with tf.device('/cpu:0'):
                self.load_or_create_model()
        else:
            self.load_or_create_model()

    def load_or_create_model(self):
        if not self.load_model(self.model_load_path):
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
                rnn_layer, state_c, state_h= rnn_cell(
                    self.rnn_units,
                    #, initial_state=initial_states[i])
                    return_sequences=ret_seq, return_state=True,
                    go_backwards=go_backwards)(rnn_layer, initial_state=initial_states[i])
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

        self.compile()

    def make_generator(self, gen):
        # TODO
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
        return [np.zeros((batch_size, self.rnn_units)) for _ in range(self.rnn_layers * 2)]

    def get_state(self, state):
        if state is None:
            return self.new_state()
        if len(state) == self.rnn_layers * 2 + 1:
            return state[1:]
        else:
            return state

    def predict(self, text, state=None, return_state=False, encode=True):
        #cached_prediction = self.predictions_cache.get_cached_value(text)
        #if cached_prediction is not None:
        #    return cached_prediction

        if encode:
            text_codes = self.str_codes(text)
        else:
            text_codes = text

        if not text:
            if self.direction == FORWARD:
                text_codes = [SOS]
            else:
                text_codes = [EOS]

        state = self.get_state(state)
        inp = [np.expand_dims(text_codes, axis=0)] + state
        outputs = self.model.predict(inp)
        probs, state = outputs[0][0], outputs[1:]
        if self.return_sequences:
            if self.direction == BACKWARD:
                probs = probs[0]
            if self.direction == FORWARD:
                probs = probs[-1]
        #self.predictions_cache.add_cache_value(text, probs)
        if return_state:
            return probs, state
        else:
            return probs

    def predict_seq(self, text, state=None, encode=True, include_pad=False):
        preds = []
        states = []
        if state is None:
            state = [np.zeros((1, self.rnn_units)) for _ in range(self.rnn_layers * 2)]
            if self.direction == FORWARD:
                inp = [np.array([[SOS]])] + state
            else:
                inp = [np.array([[EOS]])] + state
            state = self.model.predict(inp)

        if encode:
            text_codes = self.str_codes(text)
        else:
            text_codes = text
        text_codes = np.expand_dims(text_codes, axis=0)
        # TODO: Reverse for backwards
        for i in range(text_codes.shape[1]):
            inp = [text_codes[:, i, None]] + state
            outputs = self.model.predict(inp)
            probs, state = outputs[0][0][0], outputs[1:]
            preds.append(probs)
            states.append(state)
        if self.direction == BACKWARD:
            preds = preds[::-1]
            states = states[::-1]
        return preds, states

    def predict_likelihood(self, text, state_with_last, encode=True):
        # TODO: Handle previous predict
        if encode:
            text_codes = self.str_codes(text)
        else:
            text_codes = text
        prev_pred, state = state_with_last[0], state_with_last[1:]
        inp = [np.expand_dims(text_codes, axis=0)] + state
        outputs = self.model.predict(inp)[0][0].tolist()
        if self.direction == BACKWARD:
            preds = [prev_pred] + outputs[:1:-1]
            text_codes = text_codes[::-1]
        else:
            preds = [prev_pred] + outputs[:-1]
        assert len(outputs) == len(text_codes)
        ans = 0
        for i in range(len(outputs) - 1, -1, -1):
            p = preds[i][text_codes[i]]
            ans = p * (ans + 1)
        return ans

    def predict_char(self, text, char):
        return self.predict(text)[self.char_code(char)]

    def predict_end(self, text):
        return self.predict(text)[EOS]

    def predict_start(self, text):
        return self.predict(text)[SOS]

    def predict_unknown(self, text):
        return self.predict(text)[UNK]

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
