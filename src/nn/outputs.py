from typing import Optional, Dict

import tensorflow as tf


def crossentropy_output(logits: tf.Tensor,
                        y: Optional[tf.Tensor],
                        mask: tf.Tensor) -> Dict:
    if y is None:
        loss = tf.constant(0, dtype=tf.float32)
    else:
        elementwise_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y,
            logits=logits,
        )
        masked_loss = mask * elementwise_loss
        loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
    probabilities = tf.nn.softmax(logits, axis=-1)
    return {
        "loss": loss,
        "probabilities": probabilities
    }


def sigmoidal_output(logits: tf.Tensor,
                     y: Optional[tf.Tensor],
                     mask: tf.Tensor) -> Dict:
    if y is None:
        y = tf.zeros_like(logits)
    else:
        y = tf.one_hot(y, tf.shape(logits)[-1])
    elementwise_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y,
        logits=logits
    )
    expanded_mask = tf.expand_dims(mask, axis=-1)
    masked_loss = elementwise_loss * expanded_mask
    loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
    probabilities = tf.nn.sigmoid(logits)
    return {
        "loss": loss,
        "probabilities": probabilities
    }
