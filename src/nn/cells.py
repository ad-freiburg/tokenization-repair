from typing import List

import tensorflow as tf


def multilayer_lstm_cell(units):
    return tf.contrib.rnn.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(u) for u in units]
    )


class BidirectionalLSTMCell:
    """Used for language models. Leaves a gap when concatenating forward an backward hidden states."""

    def __init__(self, units: List[int]):
        self.fwd_cell = multilayer_lstm_cell(units)
        self.bwd_cell = multilayer_lstm_cell(units)

    def __call__(self, inputs):
        outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.fwd_cell,
            cell_bw=self.bwd_cell,
            inputs=inputs,
            dtype=tf.float32
        )
        fwd_outputs = outputs[0][:, :-2, :]
        bwd_outputs = outputs[1][:, 2:, :]
        concatenated = tf.concat((fwd_outputs, bwd_outputs), 2)
        return concatenated
