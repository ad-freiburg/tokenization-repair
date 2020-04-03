import tensorflow as tf

from src.models.estimator.estimator_model import EstimatorModel, log_metrics
from src.tensorflow.cells import UnidirectionalLSTMCell


class BidirectionalEstimatorSpecification:
    def __init__(self, recurrent_units, dense_units, dim, name, sigmoidal):
        self.recurrent_units = recurrent_units
        self.dense_units = dense_units
        self.dim = dim
        self.name = name
        self.sigmoidal = sigmoidal

    def __str__(self):
        return "BidirectionalEstimatorSpecification(%s, %s, %i, '%s', %s)" % (
            str(self.recurrent_units),
            str(self.dense_units),
            self.dim,
            self.name,
            str(self.sigmoidal)
        )

    def __repr__(self):
        return self.__str__()


class BidirectionalEstimator(EstimatorModel):
    def model_function(self, features, labels, mode, params: BidirectionalEstimatorSpecification):
        # inputs
        x = features["x"]  # len == seq_len + 2
        batch_size = tf.shape(x)[0]
        input_len = tf.shape(x)[1]
        encoded = tf.one_hot(x, params.dim)

        # recurrent parts
        with tf.variable_scope("fwd"):
            fwd_cell = UnidirectionalLSTMCell(params.recurrent_units)
            fwd_hidden_states = fwd_cell(encoded[:, :-1, :], batch_size)  # len == seq_len + 1
        with tf.variable_scope("bwd"):
            bwd_cell = UnidirectionalLSTMCell(params.recurrent_units)
            bwd_hidden_states = bwd_cell(encoded[:, -1:0:-1, :], batch_size)[:, ::-1, :]  # dito

        # prediction concat
        slices_prediction = (fwd_hidden_states[:, :-1, :],  # len == seq_len
                             bwd_hidden_states[:, 1:, :])  # dito
        concat_prediction = tf.concat(slices_prediction, axis=-1)

        # prediction dense part
        activation = concat_prediction
        for layer, hidden_units in enumerate(params.dense_units):
            activation = tf.layers.dense(activation, hidden_units, name="dense%i" % layer, activation=tf.nn.relu)
        logits = tf.layers.dense(activation, params.dim, name="logits")
        predictions = tf.argmax(logits, axis=-1)

        # insertion dense part
        if mode == tf.estimator.ModeKeys.PREDICT or params.sigmoidal:
            slices_insertion = (fwd_hidden_states,
                                bwd_hidden_states)
            concat_insertion = tf.concat(slices_insertion, axis=-1)
            activation_insertion = concat_insertion
            for layer, hidden_units in enumerate(params.dense_units):
                activation_insertion = tf.layers.dense(activation_insertion, hidden_units, name="dense%i" % layer,
                                                       activation=tf.nn.relu, reuse=True)
            logits_insertion = tf.layers.dense(activation_insertion, params.dim, name="logits", reuse=True)

        # training
        if mode == tf.estimator.ModeKeys.TRAIN:
            y = x[:, 1:-1]
            y_encoded = encoded[:, 1:-1, :]  # len == seq_len
            if params.sigmoidal:
                positive_examples_loss_elementwise = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                                             labels=y_encoded)
                positive_examples_loss = tf.reduce_mean(positive_examples_loss_elementwise)
                y_negative = tf.zeros((batch_size, input_len - 1, params.dim), dtype=tf.float32)
                negative_examples_loss_elementwise = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_insertion,
                                                                                             labels=y_negative)
                negative_examples_loss = tf.reduce_mean(negative_examples_loss_elementwise)
                loss = 0.5 * positive_examples_loss + 0.5 * negative_examples_loss
            else:
                elementwise_loss = tf.losses.sparse_softmax_cross_entropy(y, logits)
                loss = tf.reduce_mean(elementwise_loss)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            training_hooks = [log_metrics(loss, logits, predictions, y, None, optimizer._lr)]
            if params.sigmoidal:
                losses_hook = tf.train.LoggingTensorHook({"loss_pos": positive_examples_loss,
                                                          "loss_neg": negative_examples_loss},
                                                         every_n_iter=1)
                tf.summary.scalar("loss_pos", positive_examples_loss)
                tf.summary.scalar("loss_neg", negative_examples_loss)
                training_hooks.append(losses_hook)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=training_hooks)

        # prediction
        elif mode == tf.estimator.ModeKeys.PREDICT:
            if params.sigmoidal:
                probabilities = tf.sigmoid(logits)
                insertion_probabilities = tf.sigmoid(logits_insertion)
            else:
                probabilities = tf.nn.softmax(logits, axis=-1)
                insertion_probabilities = tf.nn.softmax(logits_insertion)
            outputs = {"probabilities": probabilities,
                       "predictions": predictions,
                       "insertion_probabilities": insertion_probabilities,
                       "labels": features["x"]}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=outputs)

    def serving_input_receiver_function(self):
        inputs = {"x": tf.placeholder(dtype=tf.int32, shape=[1, None], name="x")}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    @staticmethod
    def _output_dict(predict_fn_result):
        return {"probabilities": predict_fn_result["probabilities"][0, :, :],
                "insertion_probabilities": predict_fn_result["insertion_probabilities"][0, :, :],
                "labels": predict_fn_result["labels"][0, :]}
