from typing import Optional, List, Tuple, Dict

import numpy as np
import tensorflow as tf

from src.encoding.character_encoder import CharacterEncoder
from src.noise.token_typo_inducer import TokenTypoInducer
from src.datasets.wikipedia import Wikipedia
from src.settings import symbols
from src.sequence.functions import get_space_positions_in_merged


def x_y_training_example(sequence: str, encoder: CharacterEncoder) -> Tuple[List[int], List[int]]:
    labels = [symbols.SOS] + list(sequence) + [symbols.EOS]
    encoded = [encoder.encode_char(label) for label in labels]
    x = [encoded[i] for i in range(1, len(encoded)) if labels[i] != ' ']
    y = [encoded[i] for i in range(len(labels) - 1) if labels[i + 1] != ' ']
    return x, y


def labeling_training_example(sequence: str, encoder: CharacterEncoder) -> Tuple[np.ndarray, np.ndarray]:
    merged = sequence.replace(' ', '')
    x = encoder.encode_sequence(merged)
    y = np.zeros(len(x) - 1, dtype=int)
    spaces = get_space_positions_in_merged(sequence)
    for space_position in spaces:
        y[space_position] = 1
    return x, y


class RobustDataFnProvider:
    def __init__(self,
                 encoder: CharacterEncoder,
                 batch_size: int,
                 max_len: int,
                 start_batch: int = 0,
                 noise_prob: Optional[float] = None,
                 seed: Optional[int] = None,
                 labeling_output: bool = False):
        self.encoder = encoder
        self.batch_size = batch_size
        self.start_batch = start_batch
        self.max_len = max_len
        self.noise = noise_prob is not None
        if self.noise:
            self.corruptor = TokenTypoInducer(noise_prob, seed)
        self.labeling_output = labeling_output

    def get_batches(self) -> List[str]:
        batch = []
        for sequence in Wikipedia.training_sequences():
            if self.noise:
                sequence, _ = self.corruptor.corrupt(sequence)
            batch.append(sequence)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def encode_batch_x_y(self, batch: List[str]) -> Dict:
        batch_size = len(batch)
        examples = [x_y_training_example(sequence, self.encoder) for sequence in batch]
        batch_max_len = max(len(x) for x, _ in examples)
        batch_len = min(batch_max_len, self.max_len)
        x = np.zeros((batch_size, batch_len), dtype=int)
        y = np.zeros_like(x, dtype=int)
        mask = np.zeros_like(y, dtype=int)
        for i, (x_seq, y_seq) in enumerate(examples):
            seq_len = len(x_seq)
            if seq_len > batch_len:
                x_seq = x_seq[-batch_len:]
                y_seq = y_seq[-batch_len:]
                seq_len = batch_len
            x[i, -seq_len:] = x_seq
            y[i, -seq_len:] = y_seq
            mask[i, -seq_len:] = 1
        sequence_lengths = [len(x) for x, _ in examples]
        return {"x": x,
                "y": y,
                "sequence_lengths": sequence_lengths,
                "mask": mask}

    def encode_batch_labeling(self, batch: List[str]):
        n = len(batch)
        examples = [labeling_training_example(sequence, self.encoder) for sequence in batch]
        max_len = max(len(x) for x, _ in examples)
        batch_len = min(max_len, self.max_len)
        X = np.zeros((n, batch_len), dtype=int)
        Y = np.zeros((n, batch_len - 1), dtype=int)
        mask = np.zeros_like(Y, dtype=int)
        for i, (x, y) in enumerate(examples):
            if len(x) > self.max_len:
                x = x[:self.max_len]
                y = y[:(self.max_len - 1)]
            X[i, :len(x)] = x
            Y[i, :len(y)] = y
            mask[i, :len(y)] = 1
        return {"x": X,
                "y": Y,
                "mask": mask}

    def encode_batch(self, batch: List[str]):
        if self.labeling_output:
            return self.encode_batch_labeling(batch)
        else:
            return self.encode_batch_x_y(batch)

    def training_batch_generator(self):
        for b_i, batch in enumerate(self.get_batches()):
            if b_i < self.start_batch:
                continue
            encoded = self.encode_batch(batch)
            yield encoded

    def train_input_function(self):
        feature_shapes = {"x": tf.TensorShape((None, None)),
                          "y": tf.TensorShape((None, None)),
                          "mask": tf.TensorShape((None, None))}
        feature_types = {"x": tf.int32,
                         "y": tf.float32 if self.labeling_output else tf.int32,
                         "mask": tf.float32}
        if not self.labeling_output:
            feature_shapes["sequence_lengths"] = tf.TensorShape(None)
            feature_types["sequence_lengths"] = tf.int32
        return tf.data.Dataset.from_generator(self.training_batch_generator,
                                              output_types=feature_types,
                                              output_shapes=feature_shapes)
