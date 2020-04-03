import tensorflow as tf

from src.models.estimator.estimator_model import EstimatorModel, log_metrics
from src.tensorflow.cells import UnidirectionalLSTMCell


class RecurrentEstimatorSpecification:
    def __init__(self, recurrent_units, dense_units, dim, name, backward=False):
        self.recurrent_units = recurrent_units
        self.dense_units = dense_units
        self.dim = dim
        self.name = name
        self.backward = backward

    def __str__(self):
        return "RecurrentEstimatorSpecification(%s, %s, %i, '%s', %s)" % (
            str(self.recurrent_units),
            str(self.dense_units),
            self.dim,
            self.name,
            str(self.backward)
        )

    def __repr__(self):
        return self.__str__()


class RecurrentEstimator(EstimatorModel):
    def model_function(self,
                       features: dict,
                       labels: tf.Tensor,
                       mode: tf.estimator.ModeKeys,
                       params: RecurrentEstimatorSpecification):
        # inputs
        labels = features["x"]
        if params.backward:
            labels = labels[:, ::-1]
        if mode == tf.estimator.ModeKeys.TRAIN:
            x = labels[:, :-1]
            y = labels[:, 1:]
        else:
            x = labels
        batch_size = tf.shape(x)[0]
        encoded = tf.one_hot(x, params.dim)

        # recurrent part
        lstm_cell = UnidirectionalLSTMCell(params.recurrent_units)
        rnn_output = lstm_cell(encoded, batch_size)

        # dense_part
        activation = rnn_output
        for layer, hidden_units in enumerate(params.dense_units):
            activation = tf.layers.dense(activation, hidden_units, activation=tf.nn.relu)

        # output layer
        logits = tf.layers.dense(activation, params.dim)
        predictions = tf.argmax(logits, axis=-1)

        # training
        if mode == tf.estimator.ModeKeys.TRAIN:
            elementwise_loss = tf.losses.sparse_softmax_cross_entropy(y, logits)
            loss = tf.reduce_mean(elementwise_loss)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            training_hook = log_metrics(loss, logits, predictions, y, None, optimizer._lr)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=[training_hook])

        # prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            if params.backward:
                logits = logits[:, ::-1, :]
                predictions = predictions[:, ::-1]
            probabilities = tf.nn.softmax(logits)
            outputs = {"probabilities": probabilities,
                       "predictions": predictions,
                       "labels": features["x"]}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=outputs)

    def serving_input_receiver_function(self):
        inputs = {"x": tf.placeholder(dtype=tf.int32, shape=[1, None], name="x")}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    @staticmethod
    def _output_dict(predict_fn_result):
        return {"probabilities": predict_fn_result["probabilities"][0, :, :],
                "labels": predict_fn_result["labels"][0, :]}
