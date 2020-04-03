import tensorflow as tf


class UnidirectionalLSTMCell:
    def __init__(self, units):
        self.lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(u) for u in units]
        )

    def __call__(self, inputs, batch_size):
        initial_hidden_state = self.lstm_cell.zero_state(batch_size, tf.float32)
        rnn_output, hidden_state = tf.nn.dynamic_rnn(
            cell=self.lstm_cell,
            inputs=inputs,
            initial_state=initial_hidden_state
        )
        return rnn_output
