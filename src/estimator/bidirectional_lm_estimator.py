from typing import List, Dict, Optional

import tensorflow as tf

from src.estimator.estimator_model import EstimatorModel
from src.nn.cells import multilayer_lstm_cell
from src.nn.outputs import crossentropy_output, sigmoidal_output
from src.tensorflow.metrics import accuracy


class BidirectionalLMEstimatorSpecification:
    def __init__(self,
                 embedding: bool,
                 embedding_dim: int,
                 recurrent_units: List[int],
                 dense_units: List[int],
                 dim: int,
                 sigmoidal: bool,
                 name: str):
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.recurrent_units = recurrent_units
        self.dense_units = dense_units
        self.dim = dim
        self.sigmoidal = sigmoidal
        self.name = name

    def __str__(self):
        return "BidirectionalLMEstimatorSpecification(recurrent_units=%s, dim=%i, embedding=%s(%i), name=%s)" % (
            str(self.recurrent_units),
            self.dim,
            self.embedding,
            self.embedding_dim,
            self.name
        )


class BidirectionaLLMEstimator(EstimatorModel):
    def output(self,
               logits: tf.Tensor,
               y: Optional[tf.Tensor],
               mask: tf.Tensor) -> Dict:
        if self.specification.sigmoidal:
            return sigmoidal_output(logits, y, mask)
        else:
            return crossentropy_output(logits, y, mask)

    def model_function(self, features, labels, mode, params: BidirectionalLMEstimatorSpecification):
        x = features["x"]
        y = x[:, 1:-1]
        mask = features["mask"][:, 1:-1]

        # embedding or one-hot
        if params.embedding:
            embedding_layer = tf.keras.layers.Embedding(params.dim, params.embedding_dim)
            embedded = embedding_layer(x)
        else:
            embedded = tf.one_hot(x, params.dim)

        # fwd recurrent
        fwd_cell = multilayer_lstm_cell(params.recurrent_units)
        with tf.variable_scope("fwd"):
            fwd_outputs, fwd_hidden_states = tf.nn.dynamic_rnn(
                cell=fwd_cell,
                inputs=embedded,
                dtype=tf.float32,
            )

        # bwd recurrent
        bwd_cell = multilayer_lstm_cell(params.recurrent_units)
        with tf.variable_scope("bwd"):
            bwd_outputs, bwd_hidden_states = tf.nn.dynamic_rnn(
                cell=bwd_cell,
                inputs=embedded[:, ::-1, :],
                dtype=tf.float32,
            )
        bwd_outputs = bwd_outputs[:, ::-1, :]

        concatenated = tf.concat((fwd_outputs[:, :-2, :], bwd_outputs[:, 2:, :]), axis=2)

        # dense network: prediction
        with tf.variable_scope("dense"):
            values = concatenated
            for layer, units in enumerate(params.dense_units):
                values = tf.layers.dense(values, units, activation=tf.nn.relu, name="dense_%i" % layer)

            # output layer
            logits = tf.layers.dense(values, params.dim, name="logits")
            prediction_output = self.output(
                logits=logits,
                y=y,
                mask=mask
            )

        # same dense network: insertion
        concatenated_insertion = tf.concat((fwd_outputs[:, :-1, :], bwd_outputs[:, 1:, :]), axis=2)

        values = concatenated_insertion
        with tf.variable_scope("dense", reuse=True):
            for layer, units in enumerate(params.dense_units):
                values = tf.layers.dense(values, units, activation=tf.nn.relu, name="dense_%i" % layer)
            insertion_logits = tf.layers.dense(values, params.dim, name="logits")
            insertion_output = self.output(
                logits=insertion_logits,
                y=None,
                mask=features["insertion_mask"]
            )

        # training
        if mode == tf.estimator.ModeKeys.TRAIN:
            # loss
            prediction_loss = prediction_output["loss"]
            insertion_loss = insertion_output["loss"]
            loss = prediction_loss + insertion_loss

            # accuracy
            acc = accuracy(logits, y, padding_mask=mask)

            # optimizer
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            # metrics and logging
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("prediction_loss", prediction_loss)
            tf.summary.scalar("insertion_loss", insertion_loss)
            tf.summary.scalar("accuracy", acc)
            logging_hooks = [
                tf.train.LoggingTensorHook({"accuracy": acc,
                                            "prediction_loss": prediction_loss,
                                            "insertion_loss": insertion_loss},
                                           every_n_iter=1)
            ]

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=logging_hooks)

        # prediction
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = tf.argmax(logits, axis=-1, name="predictions")
            probabilities = prediction_output["probabilities"]
            insertion_probabilities = insertion_output["probabilities"]

            outputs = {"predictions": predictions,
                       "probabilities": probabilities,
                       "insertion_probabilities": insertion_probabilities,
                       "ground_truth": y}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=outputs)

    def serving_input_receiver_function(self):
        inputs = {"x": tf.placeholder(dtype=tf.int32, shape=[1, None], name="x"),
                  "mask": tf.placeholder(dtype=tf.float32, shape=[1, None], name="mask"),
                  "insertion_mask": tf.placeholder(dtype=tf.float32, shape=[1, None], name="insertion_mask")}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def _input_dict(self, encoded_sequence):
        return {"x": [encoded_sequence]}

    def _output_dict(self, encoded_sequence, predict_fn_result):
        return {"predictions": predict_fn_result["predictions"][0, :],
                "probabilities": predict_fn_result["probabilities"][0, :, :],
                "insertion_probabilities": predict_fn_result["insertion_probabilities"][0, :, :],
                "labels": encoded_sequence[1:-1]}
