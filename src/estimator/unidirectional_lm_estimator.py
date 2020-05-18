from typing import List

import numpy as np
import tensorflow as tf

from src.estimator.estimator_model import EstimatorModel
from src.nn.cells import multilayer_lstm_cell
from src.nn.outputs import crossentropy_output
from src.tensorflow.metrics import accuracy


class UnidirectionalLMEstimatorSpecification:
    def __init__(self,
                 backward: bool,
                 embedding: bool,
                 embedding_dim: int,
                 recurrent_units: List[int],
                 dense_units: List[int],
                 dim: int,
                 name: str,
                 x_and_y_input: bool = False):
        self.backward = backward
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.recurrent_units = recurrent_units
        self.dense_units = dense_units
        self.dim = dim
        self.name = name
        self.x_and_y_input = x_and_y_input

    def gets_x_and_y_input(self):
        return hasattr(self, "x_and_y_input") and self.x_and_y_input

    def __str__(self):
        return "UnidirectionalLMEstimatorSpecification(recurrent_units=%s, dim=%i, embedding=%s(%i), name=%s)" % (
            str(self.recurrent_units),
            self.dim,
            self.embedding,
            self.embedding_dim,
            self.name
        )


class UnidirectionalLMEstimator(EstimatorModel):
    def _hidden_state_tensor(self, features):
        """
        Subroutine of the model function to transforms the placeholders refering to the RNN cell's hidden state
        into a tuple of LSTMStateTuples, which can be used to call tf.nn.dynamic_run.
        :param features: dictionary of placeholders as used in the model function.
        :return: tuple of LSTMStateTuples
        """
        n_layers = len(self.specification.recurrent_units)
        layer_hidden_states = []
        for layer in range(n_layers):
            c_name, h_name = self._hidden_state_tensor_names(layer)
            hidden_state_tensor = tf.nn.rnn_cell.LSTMStateTuple(c=features[c_name],
                                                                h=features[h_name])
            layer_hidden_states.append(hidden_state_tensor)
        layer_hidden_states = tuple(layer_hidden_states)
        return layer_hidden_states

    @staticmethod
    def _hidden_state_tensor_names(layer):
        """
        Names for the c-state and h-state of layer K.
        :param layer: K
        :return: name of c-state, name of h-state
        """
        return "state.%i.c" % layer, "state.%i.h" % layer

    def hidden_state_placeholders(self):
        """
        Creates a dictionary of placeholders to feed all layers' hidden states.
        The placeholders' dimensions are defined in the specification.
        :return: dictionary containing placeholders
            keys "state.K.c" and "state.K.h" refer to layer K's c-state and h-state
        """
        n_layers = len(self.specification.recurrent_units)
        layer_hidden_state_placeholders = {}
        for layer in range(n_layers):
            units = self.specification.recurrent_units[layer]
            c_name, h_name = self._hidden_state_tensor_names(layer)
            layer_hidden_state_placeholders[c_name] = tf.placeholder(dtype=tf.float32, shape=[None, units], name=c_name)
            layer_hidden_state_placeholders[h_name] = tf.placeholder(dtype=tf.float32, shape=[None, units], name=h_name)
        return layer_hidden_state_placeholders

    def _initial_cell_state(self):
        """
        Creates a dictionary of zero-vectors for each layer's hidden state.
        Use for the first step when predicting stepwise.
        Assumes batch_size = 1 during prediction.
        :return: dictionary of zero-vectors
            keys "state.K.c" and "state.K.h" refer to layer K's c-state and h-state
        """
        n_layers = len(self.specification.recurrent_units)
        state_dict = {}
        for layer in range(n_layers):
            units = self.specification.recurrent_units[layer]
            c_name, h_name = self._hidden_state_tensor_names(layer)
            state_dict[c_name] = np.zeros((1, units))
            state_dict[h_name] = np.zeros((1, units))
        return state_dict

    def _hidden_state_from_prediction_result(self, result):
        """
        Collects the vectors refering the model's recurrent cell's hidden states.
        Used in stepwise prediction to collect the cell state for the state dictionary.
        :param result: dictionary as returned by self.predict_fn
        :return: dictionary containing hidden states
            keys "state.K.c" and "state.K.h" refer to layer K's hidden state
        """
        hidden_state = {}
        for layer in range(len(self.specification.recurrent_units)):
            c_name, h_name = self._hidden_state_tensor_names(layer)
            hidden_state[c_name] = result[c_name]
            hidden_state[h_name] = result[h_name]
        return hidden_state

    def model_function(self, features, labels, mode, params: UnidirectionalLMEstimatorSpecification):
        if params.gets_x_and_y_input():
            x = features["x"]
            y = features["y"] if mode == tf.estimator.ModeKeys.TRAIN else None
            mask = features["mask"] if mode == tf.estimator.ModeKeys.TRAIN else None
            if params.backward:
                x = x[:, ::-1]
                y = y[:, ::-1] if y is not None else None
                mask = mask[:, ::-1] if mask is not None else None
        else:
            labels = features["x"]  # (n x m)

            if params.backward:
                labels = labels[:, ::-1]

            if mode == tf.estimator.ModeKeys.TRAIN:
                mask = features["mask"]  # (n x m)
                if params.backward:
                    mask = mask[:, ::-1]
                x = labels[:, :-1]
                y = labels[:, 1:]
                mask = mask[:, 1:]
            else:
                x = labels
                y = None
                mask = None

        sequence_lengths = features["sequence_lengths"]
        batch_size = tf.shape(x)[0]

        # embedding or one-hot
        if params.embedding:
            embedding_layer = tf.keras.layers.Embedding(params.dim, params.embedding_dim)
            embedded = embedding_layer(x)
        else:
            embedded = tf.one_hot(x, params.dim)

        # lstm cell
        cell = multilayer_lstm_cell(params.recurrent_units)

        # initial hidden state
        if mode == tf.estimator.ModeKeys.TRAIN:
            initial_hidden_state = cell.zero_state(batch_size, tf.float32)
        else:
            initial_hidden_state = self._hidden_state_tensor(features)

        # recurrent execution
        recurrent_outputs, hidden_states = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=embedded,
            initial_state=initial_hidden_state,
            dtype=tf.float32,
            sequence_length=sequence_lengths
        )

        # dense network
        values = recurrent_outputs
        for layer, units in enumerate(params.dense_units):
            values = tf.layers.dense(values, units, activation=tf.nn.relu, name="dense_%i" % layer)

        # output layer
        logits = tf.layers.dense(values, params.dim)

        # training
        if mode == tf.estimator.ModeKeys.TRAIN:
            output = crossentropy_output(logits, y, mask)

            # loss
            loss = output["loss"]

            # accuracy
            acc = accuracy(logits, y, padding_mask=mask)

            # optimizer
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            # metrics and logging
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accuracy", acc)
            logging_hooks = [
                tf.train.LoggingTensorHook({"accuracy": acc}, every_n_iter=1),
                self.training_result_hook("loss", loss),
                self.training_result_hook("accuracy", acc)
            ]

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=logging_hooks)

        # prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            probabilities = tf.nn.softmax(logits, axis=-1)
            predictions = tf.argmax(logits, axis=-1, name="predictions")
            if params.backward:
                probabilities = probabilities[:, ::-1, :]
                predictions = predictions[:, ::-1]
            outputs = {"predictions": predictions,
                       "probabilities": probabilities}
            for layer in range(len(params.recurrent_units)):
                c_name, h_name = self._hidden_state_tensor_names(layer)
                outputs[c_name] = hidden_states[layer].c
                outputs[h_name] = hidden_states[layer].h
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=outputs)

    def serving_input_receiver_function(self):
        inputs = {"x": tf.placeholder(dtype=tf.int32, shape=[None, None], name="x"),
                  "mask": tf.placeholder(dtype=tf.float32, shape=[None, None], name="mask"),
                  "insertion_mask": tf.placeholder(dtype=tf.float32, shape=[None, None], name="insertion_mask"),
                  "sequence_lengths": tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_lengths")}
        state = self.hidden_state_placeholders()
        for name in state:
            inputs[name] = state[name]
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def _input_dict(self, encoded_sequence):
        input_dict = self._initial_cell_state()
        x = encoded_sequence[1:] if self.specification.backward else encoded_sequence[:-1]
        input_dict["x"] = [x]
        input_dict["sequence_lengths"] = [len(x)]
        return input_dict

    def _output_dict(self, encoded_sequence, predict_fn_result):
        return {"predictions": predict_fn_result["predictions"][0, :],
                "probabilities": predict_fn_result["probabilities"][0, :, :],
                "labels": encoded_sequence}

    def initial_state(self):
        return {
            "cell_state": self._initial_cell_state(),
            "sequence": []
        }

    def step(self, state, x, include_sequence=True):
        input_dict = state["cell_state"]
        input_dict["x"] = [[x]]
        input_dict["sequence_lengths"] = [1]
        result = self.predict_fn(input_dict=input_dict)
        new_state = {
            "cell_state": self._hidden_state_from_prediction_result(result),
            "probabilities": result["probabilities"][0, 0, :]
        }
        if include_sequence:
            new_state["sequence"] = state["sequence"] + [x]
        return new_state

    def step_batch(self, states, x_vec):
        n = len(states)
        x_matrix = [[x] for x in x_vec]
        input_dict = {"x": x_matrix,
                      "sequence_lengths": [1 for _ in range(n)]}
        for layer in range(len(self.specification.recurrent_units)):
            c_name, h_name = self._hidden_state_tensor_names(layer)
            input_dict[c_name] = [state["cell_state"][c_name][0] for state in states]
            input_dict[h_name] = [state["cell_state"][h_name][0] for state in states]
        result = self.predict_fn(input_dict)
        states = []
        for i in range(n):
            state = {"probabilities": result["probabilities"][i, -1, :],
                     "cell_state": {}}
            for layer in range(len(self.specification.recurrent_units)):
                c_name, h_name = self._hidden_state_tensor_names(layer)
                state["cell_state"][c_name] = [result[c_name][i]]
                state["cell_state"][h_name] = [result[h_name][i]]
            states.append(state)
        return states
