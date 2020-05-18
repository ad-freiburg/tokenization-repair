from typing import List

import tensorflow as tf

from src.estimator.estimator_model import EstimatorModel
from src.nn.cells import multilayer_lstm_cell
from src.tensorflow.metrics import accuracy


class BidirectionalLabelingEstimatorSpecification:
    def __init__(self,
                 recurrent_units: List[int],
                 dense_units: List[int],
                 dim: int,
                 name: str):
        self.recurrent_units = recurrent_units
        self.dense_units = dense_units
        self.dim = dim
        self.name = name


class BidirectionalLabelingEstimator(EstimatorModel):
    def model_function(self, features, labels, mode, params):
        x = features["x"]

        x = tf.one_hot(x, params.dim)

        # fwd recurrent
        fwd_cell = multilayer_lstm_cell(params.recurrent_units)
        with tf.variable_scope("fwd"):
            fwd_outputs, fwd_hidden_states = tf.nn.dynamic_rnn(
                cell=fwd_cell,
                inputs=x[:, :-1, :],
                dtype=tf.float32,
            )

        # bwd recurrent
        bwd_cell = multilayer_lstm_cell(params.recurrent_units)
        with tf.variable_scope("bwd"):
            bwd_outputs, bwd_hidden_states = tf.nn.dynamic_rnn(
                cell=bwd_cell,
                inputs=x[:, :0:-1, :],
                dtype=tf.float32,
            )
        bwd_outputs = bwd_outputs[:, ::-1, :]

        values = tf.concat((fwd_outputs, bwd_outputs), axis=-1)

        with tf.variable_scope("dense"):
            for layer, units in enumerate(params.dense_units):
                values = tf.layers.dense(values, units, activation=tf.nn.relu, name="dense_%i" % layer)

        logits = tf.layers.dense(values, 1)[:, :, 0]
        probabilities = tf.sigmoid(logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            y = features["y"]
            mask = features["mask"]

            loss = tf.losses.mean_squared_error(y, probabilities, weights=mask)
            print(probabilities)
            print(y)
            print(loss)

            # accuracy
            predictions = tf.cast(probabilities >= 0.5, dtype=tf.int32)
            correct = tf.equal(predictions, tf.cast(y, dtype=tf.int32))
            correct = tf.cast(correct, dtype=tf.float32)
            acc = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            # metrics and logging
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accuracy", acc)
            #logging_hooks = []
            logging_hooks = [
                tf.train.LoggingTensorHook({"accuracy": acc},
                                           every_n_iter=1)
            ]

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=logging_hooks)

        else:
            outputs = {"probabilities": probabilities}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=outputs)

    def serving_input_receiver_function(self):
        inputs = {"x": tf.placeholder(dtype=tf.int32, shape=[None, None], name="x")}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def _input_dict(self, encoded_sequence):
        return {"x": [encoded_sequence]}

    def _output_dict(self, encoded_sequence, predict_fn_result):
        return {"probabilities": predict_fn_result["probabilities"][0, :]}