import tensorflow as tf
import numpy as np

import project
from src.datasets.wikipedia import Wikipedia
from src.encoding.character_encoder import get_encoder


MAX_LEN = 256


class LabelingModel(tf.keras.Model):
    def __init__(self, units: int, v_size: int):
        super(LabelingModel, self).__init__()
        self.v_size = v_size
        self.fwd_rnn = tf.keras.layers.LSTM(units, return_sequences=True)
        self.bwd_rnn = tf.keras.layers.LSTM(units, return_sequences=True)
        self.dense = tf.keras.layers.Dense(units, activation="relu")
        self.out = tf.keras.layers.Dense(2)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.int32)
        one_hot = tf.one_hot(inputs, depth=self.v_size)
        fwd_states = self.fwd_rnn(one_hot)
        bwd_states = self.bwd_rnn(one_hot[:, ::-1, :])
        bwd_states = bwd_states[:, ::-1, :]
        states_concatenated = tf.concat((fwd_states, bwd_states), axis=-1)
        hidden_activation = self.dense(states_concatenated)
        logits = self.out(hidden_activation)
        return logits


def encode_batch(sequences, encoder):
    max_len = max(len(sequence) for sequence in sequences)
    max_len = min(max_len, MAX_LEN)
    X = np.zeros(shape=(len(sequences), max_len), dtype=int)
    for s_i, sequence in enumerate(sequences):
        sequence = sequence[:MAX_LEN]
        encoded = encoder.encode_sequence(sequence)
        X[s_i, :len(sequence)] = encoded[1:-1]
    return X


def get_y(sequences):
    max_len = max(len(sequence) for sequence in sequences)
    max_len = min(max_len, MAX_LEN)
    Y = np.zeros(shape=(len(sequences), max_len, 2), dtype=int)
    for s_i, sequence in enumerate(sequences):
        sequence = sequence[:MAX_LEN]
        is_space = [1 if char == ' ' else 0 for char in sequence]
        Y[s_i, :len(sequence), 0] = is_space
        Y[s_i, :len(sequence), 1] = np.ones_like(is_space) - is_space
    return Y


if __name__ == "__main__":
    encoder = get_encoder(100)

    model = LabelingModel(units=1, v_size=encoder.dim())
    model.compile(optimizer="sgd",
                  loss=tf.losses.sigmoid_cross_entropy)
    print(model)

    sequences = Wikipedia.training_sequences()

    for b_i in range(10000):
        batch = []
        for _ in range(32):
            batch.append(next(sequences))
        X = encode_batch(batch, encoder)
        Y = get_y(batch)
        print(X.shape)
        print(Y.shape)
        result = model.predict(X)
        print(result.shape)
        model.fit(x=X, y=Y)