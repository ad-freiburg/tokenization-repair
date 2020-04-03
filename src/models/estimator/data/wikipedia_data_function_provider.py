import tensorflow as tf
import numpy as np

from src.data.wikipedia import Wikipedia


class WikipediaDataFunctionProvider:
    """
    Provides a data function that iterates over batches of encoded sequences from Wikipedia.
    """

    def __init__(self, encoder, batch_size, max_seq_len, start_batch=0):
        """
        Constructor.

        :param encoder: supports a encode_sequence function that transforms a sequence into a list of labels
        :param batch_size: number of sequences per batch
        :param max_seq_len: encoded sequences get cut at this length
        """
        self.encoder = encoder
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.start_batch = start_batch

    def _encode_batch(self, sequences):
        return [self.encoder.encode_sequence(sequence)[:self.max_seq_len] for sequence in sequences]

    def _training_batch_generator(self):
        if self.start_batch > 0:
            print("Skipping first %i batches..." % self.start_batch)
        for i, batch in enumerate(Wikipedia.training_batches(self.batch_size)):
            if i > self.start_batch:
                encoded = np.asarray(self._encode_batch(batch.sequences))
                yield {"x": encoded}

    def train_input_function(self):
        feature_shapes = {"x": tf.TensorShape((None, None))}
        feature_types = {"x": tf.int32}
        return tf.data.Dataset.from_generator(self._training_batch_generator,
                                              output_shapes=feature_shapes,
                                              output_types=feature_types)
