import numpy as np
from keras.layers import Layer
from utils.logger import logger
from constants import MICRO_EPS
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


class Sparse(Layer):

    def __init__(self, **kwargs):
        super(Sparse, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        from keras.constraints import MinMaxNorm, NonNeg, UnitNorm
        assert len(input_shape) == 2, 'Rank must be 1'
        from keras.initializers import Zeros, RandomNormal, glorot_uniform
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],),
                                      # initializer='he_normal',
                                      initializer=RandomNormal(1.0, 0.01),  # 1 / np.sqrt(input_shape[1])),
                                      # constraint=NonNeg(),
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[1],),
                                    initializer=Zeros(),
                                    trainable=True)

        super(Sparse, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x * self.kernel + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class Onehot(Layer):
    def __init__(self, classes, *args, **kwargs):
        super(Onehot, self).__init__(*args, **kwargs)
        self.classes = classes

    def get_config(self):
        dic = super(Onehot, self).get_config()
        dic['classes'] = self.classes
        return dic

    def call(self, x):
        import tensorflow as tf
        return tf.one_hot(tf.cast(x, tf.int32), self.classes)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.classes,)


def SeqLikelihood(seq_model, history_length, name='', reverse=False, lf=5):
    from keras.layers import Lambda
    name = 'sequence_log_likelihood_' + name

    def bayes(seq):
        import tensorflow as tf
        seqi = seq
        if reverse:
            seqi = seq[:, -history_length - lf:]
        else:
            seqi = seq[:, : history_length + lf]

        logits = seq_model(seqi)
        # logits = tf.nn.softmax(logits)
        logits = tf.log(logits + 1e-18)
        idx = tf.cast(tf.expand_dims(seqi, axis=-1), tf.int32)
        if reverse:
            logits = tf.squeeze(tf.batch_gather(logits[:, 1:], idx[:, :-1]), axis=-1)
        else:
            logits = tf.squeeze(tf.batch_gather(logits[:, :-1], idx[:, 1:]), axis=-1)
        if reverse:
            logits = logits[:, : -history_length + 1]
        else:
            logits = logits[:, history_length - 1:]
        logits = tf.cumsum(logits, axis=-1, reverse=reverse)
        return logits
        """
        if reverse:
            logits = logits / tf.expand_dims(tf.range(10, 0, -1, dtype=tf.float32), axis=0)
        else:
            logits = logits / tf.expand_dims(tf.range(1, 6, dtype=tf.float32), axis=0)
        return idx  # logits
        """
    return Lambda(bayes, name=name, dtype='int32')


def SeqLikelihoodSummary(history_length, name='', reverse=False):
    from keras.layers import Lambda
    name = 'sequence_log_likelihood_' + name

    def bayes(logits):
        import tensorflow as tf
        # logits = tf.sigmoid(logits)
        logits = tf.exp(-logits)
        logits = tf.reduce_sum(logits, axis=-1, keepdims=True)
        #logits = tf.cumsum(logits, axis=-1, reverse=reverse)
        #if reverse:
        #    logits = logits / tf.expand_dims(tf.range(32, 0, -1, dtype=tf.float32), axis=0)
        #else:
        #    logits = logits / tf.expand_dims(tf.range(1, 32 + 1, dtype=tf.float32), axis=0)
        logits = tf.log(logits + 1e-18)
        return logits

    return Lambda(bayes, name=name, dtype='int32')


def batch_scatter(alphabet_size, expand=True):
    def mselect(args):
        import keras.backend as K
        logits, top_idxs = args
        sct = K.one_hot(K.cast(top_idxs, 'int32'), alphabet_size)
        #sct = sct * (K.sqrt(K.sum(sct, axis=0, keepdims=True)) + 1e-8)
        return K.batch_dot(logits, sct, axes=1)
        """
        if expand:
            return K.batch_dot(logits, K.one_hot(K.cast(top_idxs, 'int32'),
                                                 alphabet_size), axes=1)
        else:
            return K.batch_dot(logits[:, :, 0], K.one_hot(K.cast(top_idxs, 'int32'),
                                                          alphabet_size), axes=1)
        """
    return mselect


def batch_gather():
    def mselect(args):
        import tensorflow as tf
        logits, seqi = args
        idx = tf.cast(seqi, tf.int32)
        logits = tf.batch_gather(logits, idx)
        return logits
    return mselect


def sparse_accuracy(y_true, y_pred):
    import keras.backend as K
    return K.cast(K.equal(y_true[:, :, 0],
                  K.cast(K.argmax(y_pred, axis=-1), y_true.dtype)), K.floatx())


def sparse_top_k_categorical_acc(k=5):
    def sparse_top_k_accuracy(y_true, y_pred):
        import keras.backend as K
        y_true = K.reshape(y_true, [-1])
        y_pred = K.reshape(y_pred, [-1, y_pred.shape[-1]])
        return K.in_top_k(y_pred, K.cast(y_true, 'int32'), k)
    return sparse_top_k_accuracy


def balanced_categorical_crossentropy(y_true, y_pred):
    import keras.backend as K
    y_true = K.one_hot(K.squeeze(K.cast(y_true, 'int32'), axis=-1), 101)
    weights = (K.sum(y_true, axis=0, keepdims=True)) ** 0.1 + 1
    weights = 1 - weights / K.sum(weights, keepdims=True)
    return K.sum(-K.log(y_pred + MICRO_EPS) * (y_true * weights), axis=-1)


all_custom_layers = {
    'Sparse': Sparse, 'Onehot': Onehot, 'sparse_accuracy': sparse_accuracy,
    'sparse_top_k_accuracy': sparse_top_k_categorical_acc(),
    'balanced_categorical_crossentropy': balanced_categorical_crossentropy}
