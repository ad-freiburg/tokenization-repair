from typing import List, Iterator, Dict, Optional, Tuple

import random
import math
from enum import Enum

from src.settings import paths
from src.helper.files import get_files, read_file, read_sequences
from src.helper.pickle import load_object
from src.encoding.character_encoder import CharacterEncoder


class DatasetSplit(Enum):
    TRAINING = 1
    DEVELOPMENT = 2
    TEST = 3


class Wikipedia:
    """The Wikipedia dataset."""
    @staticmethod
    def training_sequences(n_sequences: Optional[int]=None):
        """Reads the correct training sequences stored in single-file format.

        :param n_sequences: number of sequences, set None to retrieve all
        :return: iterator over paragraph texts
        """
        training_sequences = read_sequences(paths.WIKI_TRAINING_FILE)
        for i, sequence in enumerate(training_sequences):
            if n_sequences is not None and i == n_sequences:
                break
            yield sequence

    @staticmethod
    def _read_sequences(path: str,
                        n_sequences: Optional[int]=None,
                        seed: Optional[int]=None):
        sequences = list(read_sequences(path))
        if seed is not None:
            random.Random(seed).shuffle(sequences)
        if n_sequences is not None:
            sequences = sequences[:n_sequences]
        for sequence in sequences:
            yield sequence

    @staticmethod
    def development_sequences(n_sequences: Optional[int]=None,
                              seed: Optional[int]=None):
        """Reads the correct development sequences stored in single-file format.

        :param n_sequences: number of sequences, set None to retrieve all
        :param seed: seed for shuffling, set None for unshuffled
        :return: iterator over paragraph texts
        """
        return Wikipedia._read_sequences(paths.WIKI_DEVELOPMENT_FILE, n_sequences, seed)

    @staticmethod
    def test_sequences(n_sequences: Optional[int]=None,
                       seed: Optional[int]=None):
        """Reads the correct test sequences stored in single-file format.

        :param n_sequences: number of sequences, set None to retrieve all
        :param seed: seed for shuffling, set None for unshuffled
        :return: iterator over paragraph texts
        """
        return Wikipedia._read_sequences(paths.WIKI_TEST_FILE, n_sequences, seed)
