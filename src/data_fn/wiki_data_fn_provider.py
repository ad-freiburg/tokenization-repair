from typing import Dict, List, Optional, Tuple

import tensorflow as tf
import numpy as np

from src.settings import paths
from src.helper.files import read_sequences
from src.settings import symbols
from src.noise.token_typo_inducer import TokenTypoInducer
from src.encoding.character_encoder import CharacterEncoder


class WikiDataFnProvider:
    def __init__(self,
                 encoder: CharacterEncoder,
                 batch_size: int,
                 start_batch: int = 0,
                 max_len: Optional[int] = None,
                 bidirectional_mask: bool = False,
                 noise_prob: Optional[float] = None,
                 mask_noisy: bool = False,
                 pad_sos: bool = False,
                 seed: Optional[int] = None):
        self.encoder = encoder
        self.batch_size = batch_size
        self.start_batch = start_batch
        self.max_len = max_len
        self.bidirectional_mask = bidirectional_mask
        self.pad_sos = pad_sos
        self.noise = noise_prob is not None
        if self.noise:
            self.mask_noisy = mask_noisy
            self.corruptor = TokenTypoInducer(noise_prob, seed)

    def read_sequences(self):
        return read_sequences(paths.WIKI_TRAINING_FILE)

    def read_batches(self):
        batch = []
        for sequence in self.read_sequences():
            batch.append(sequence)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def encode_batch(self,
                     batch: List[str],
                     masks: Optional[List[List[int]]] = None) -> Dict:
        if masks is None:
            masks = [[1] * len(sequence) for sequence in batch]
        sos_eos_mask = [0] if self.bidirectional_mask else [1]
        n = len(batch)
        max_len = max(len(sequence) for sequence in batch) + 2
        if self.max_len is not None:
            max_len = min(max_len, self.max_len)
        padding_symbol = symbols.SOS if self.pad_sos else symbols.EOS
        X = np.ones(shape=(n, max_len), dtype=int) * self.encoder.encode_char(padding_symbol)
        mask = np.zeros(shape=(n, max_len), dtype=float)
        insertion_mask = np.zeros(shape=(n, max_len - 1), dtype=float)
        sequence_lengths = np.zeros(shape=n, dtype=int)
        for i, sequence in enumerate(batch):
            encoded = self.encoder.encode_sequence(sequence)
            length = min(len(encoded), max_len)
            sequence_lengths[i] = length
            padded_mask = (sos_eos_mask + masks[i] + sos_eos_mask)
            if self.pad_sos:
                X[i, -length:] = encoded[-length:]
                mask[i, -length:] = padded_mask[-length:]
                insertion_mask[i, -(length - 1):] = 1
            else:
                X[i, :length] = encoded[:length]
                mask[i, :length] = padded_mask[:length]
                insertion_mask[i, :(length - 1)] = 1
        return {"x": X,
                "mask": mask,
                "insertion_mask": insertion_mask,
                "sequence_lengths": sequence_lengths}

    def corrupt_batch(self, batch: List[str]) -> Tuple[List[str], List[List[int]]]:
        corrupt_sequences = []
        masks = []
        for sequence in batch:
            if self.noise:
                corrupt_sequence, mask = self.corruptor.corrupt(sequence)
                corrupt_sequences.append(corrupt_sequence)
            else:
                corrupt_sequences.append(sequence)
            if self.noise and self.mask_noisy:
                masks.append(mask)
            else:
                masks.append([1] * len(corrupt_sequences[-1]))
        return corrupt_sequences, masks

    def training_batch_generator(self):
        if self.start_batch > 0:
            print("skipping first %i batches..." % self.start_batch)
        for b_i, batch in enumerate(self.read_batches()):
            if b_i < self.start_batch:
                continue
            batch, masks = self.corrupt_batch(batch)
            encoded = self.encode_batch(batch, masks)
            print(encoded["x"].shape, encoded["mask"].shape)
            yield encoded

    def train_input_function(self):
        feature_shapes = {"x": tf.TensorShape((None, None)),
                          "mask": tf.TensorShape((None, None)),
                          "insertion_mask": tf.TensorShape((None, None)),
                          "sequence_lengths": tf.TensorShape(None)}
        feature_types = {"x": tf.int32,
                         "mask": tf.float32,
                         "insertion_mask": tf.float32,
                         "sequence_lengths": tf.int32}
        return tf.data.Dataset.from_generator(self.training_batch_generator,
                                              output_types=feature_types,
                                              output_shapes=feature_shapes)
