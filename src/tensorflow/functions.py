import tensorflow as tf


def future_lookup_prevention_mask(keys_len, query_len, prevail_val=1.0, cancel_val=0.0, include_diagonal=False):
    ones = tf.ones(keys_len * (keys_len + 1) // 2)
    zero_ones_mask = tf.ones((keys_len, keys_len)) - tf.contrib.distributions.fill_triangular(ones, upper=True)
    prevail_mask = zero_ones_mask * prevail_val
    cancel_mask = tf.contrib.distributions.fill_triangular(ones * tf.constant(cancel_val, dtype=tf.float32), upper=True)
    mask = prevail_mask + cancel_mask
    if include_diagonal:
        mask = tf.matrix_set_diag(mask, tf.ones(keys_len, dtype=tf.float32) * prevail_val)
    return mask[(-query_len):, :]


def masked_mean(tensor, mask):
    masked_tensor = tensor * mask
    masked_tensor_sum = tf.math.reduce_sum(masked_tensor)
    mask_sum = tf.math.reduce_sum(mask)
    return masked_tensor_sum / mask_sum
