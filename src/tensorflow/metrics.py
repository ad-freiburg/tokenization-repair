import tensorflow as tf


def accuracy(probabilities, y, padding_mask=None):
    if padding_mask is None:
        padding_mask = tf.ones_like(y, dtype=tf.float32)
    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
    correct = tf.math.equal(predictions, y)
    correct = tf.dtypes.cast(correct, dtype=tf.float32)
    correct = correct * padding_mask
    n_correct = tf.math.reduce_sum(correct)
    n = tf.math.reduce_sum(padding_mask)
    acc = n_correct / n
    return acc
