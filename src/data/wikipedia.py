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


class Sequence:
    """Represents a sequence in the single-file dataset format.

    Contains the sequence ID, dataset partition, byte offset
    and byte length to cut out the sequence from the file, and length in characters.
    """

    def __init__(self,
                 id: int,
                 split: DatasetSplit,
                 byte_offset: int,
                 byte_len: int,
                 char_len: int):
        """
        :param id: sequence id
        :param split: training, development or test
        :param byte_offset: start byte of the sequence in the file
        :param byte_len: end byte of the sequence in the file, inclusive \n symbol
        :param char_len: length of sequence in characters, exclusive \n symbol
        """
        self.id = id
        self.split = split
        self.byte_offset = byte_offset
        self.byte_len = byte_len
        self.char_len = char_len


class Batch:
    """A collection of training sequences of the same length (in characters)."""

    def __init__(self,
                 sequences: List[str],
                 batch_size: int,
                 sequence_length: int):
        """
        :param sequences: list of sequence texts
        :param batch_size: number of sequences
        :param sequence_length: length of sequences in characters
        """
        self.sequences = sequences
        self.batch_size = batch_size
        self.sequence_lenght = sequence_length

    def print(self):
        print(self.batch_size, self.sequence_lenght)
        for sequence in self.sequences:
            print(sequence)


class Wikipedia:
    """The Wikipedia dataset."""

    @staticmethod
    def file_iterator(benchmark_name: str="correct",
                      split: str="training"
                      ) -> Iterator[str]:
        """Iterates over the paragraph files of a given benchmark and partition.

        :param benchmark_name: name of the benchmark, equals the benchmark folder name
        :param split: name of the partition, either training, development or test
        :return: iterator over file paths, relative from the Wikipedia directory defined in src.settings.paths
        """
        dir = split + "/" + benchmark_name + "/texts/"
        subdirs = sorted(get_files(paths.WIKI_DIR + dir))
        for subdir in subdirs:
            files = sorted(get_files(paths.WIKI_DIR + dir + subdir))
            for file in files:
                yield dir + subdir + "/" + file

    @staticmethod
    def number_of_files(split: str,
                        benchmark_name: str="correct"
                        ) -> int:
        """Counts the files of a given benchmark partition.

        :param split: name of the partition, either training, development or test
        :param benchmark_name: name of the benchmark, equals the benchmark folder name
        :return:
        """
        n_files = 0
        for _ in Wikipedia.file_iterator(split=split, benchmark_name=benchmark_name):
            n_files += 1
        return n_files

    @staticmethod
    def get_sequence(file):
        """Reads the content of a paragraph file, i.e. the paragraph.

        :param file: relative path to the file from the Wikipedia directory defined in src.settings.paths
        :return: the paragraph text
        """
        return read_file(paths.WIKI_DIR + file)

    @staticmethod
    def _read_sequences(file: str,
                        sequence_list: List[Sequence],
                        seed: int,
                        n_sequences: int
                        ) -> Iterator[str]:
        """Reads the sequences from a file in the single-file format.

        :param file: absolute path to the file
        :param sequence_list: list of Sequence objects that define start end end byte of each sequence
        :param seed: seed for shuffling, set None for unshuffled
        :param n_sequences: number of sequences, set None to retrieve all sequences
        :return: iterator over paragraph texts
        """
        if seed is not None:
            random.Random(seed).shuffle(sequence_list)
        if n_sequences is not None:
            sequence_list = sequence_list[:n_sequences]
        with open(file, "rb") as f:
            for sequence in sequence_list:
                f.seek(sequence.byte_offset)
                bytes = f.read(sequence.byte_len)
                text = bytes.decode(encoding="utf8")[:-1]
                yield text

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
    def development_sequences(n_sequences: Optional[int]=None,
                              seed: Optional[int]=None):
        """Reads the correct development sequences stored in single-file format.

        :param n_sequences: number of sequences, set None to retrieve all
        :param seed: seed for shuffling, set None for unshuffled
        :return: iterator over paragraph texts
        """
        development_sequences = load_object(paths.WIKI_DEVELOPMENT_SEQUENCES)
        file = paths.WIKI_DEVELOPMENT_FILE
        return Wikipedia._read_sequences(file, development_sequences, seed, n_sequences)

    @staticmethod
    def test_sequences(n_sequences: Optional[int]=None,
                       seed: Optional[int]=None):
        """Reads the correct test sequences stored in single-file format.

        :param n_sequences: number of sequences, set None to retrieve all
        :param seed: seed for shuffling, set None for unshuffled
        :return: iterator over paragraph texts
        """
        test_sequences = load_object(paths.WIKI_TEST_SEQUENCES)
        file = paths.WIKI_TEST_FILE
        return Wikipedia._read_sequences(file, test_sequences, seed, n_sequences)

    @staticmethod
    def get_character_counts() -> Dict[str, int]:
        """Loads the character counts from disk. Path is defined in src.settings.path.

        :return: dictionary char -> count
        """
        return load_object(paths.WIKI_CHARACTER_COUNT_DICT)

    @staticmethod
    def get_encoder(n_characters: int
                    ) -> CharacterEncoder:
        """Creates an encoder for the n most frequent characters.

        The character count dictionary is read from disk.

        :param n_characters: number of characters to encode
        :return: encoder
        """
        character_counts = Wikipedia.get_character_counts()
        return CharacterEncoder(character_counts, n_characters)

    @staticmethod
    def training_batches(batch_size: int,
                         seed: int=42
                         ) -> Iterator[Batch]:
        """Iterates over batches of correct training sequences stored in split format.

        Loads the number of sequences of equal length from disk and splits the sequences into batches of the same
        length. Reads the split files in shuffled order.
        Actually, only the order of the files is shuffled, but the batches are fixed, and the order of batches of the
        same sequence lenght is fixed. For training for multiple epochs, one should really shuffle the sequences and
        batches.

        :param batch_size: maximum number of sequences per batch
        :param seed: seed to shuffle the files
        :return: iterator over batches
        """
        num_sequences = load_object(paths.WIKI_TRAINING_SEQUENCE_COUNTS)
        batches = []
        for seq_len in sorted(num_sequences):
            n_batches = math.ceil(num_sequences[seq_len] / batch_size)
            batches += [seq_len] * n_batches
        random.Random(seed).shuffle(batches)
        file_positions = {}
        for seq_len in batches:
            if seq_len in file_positions:
                start = file_positions[seq_len]
            else:
                start = 0
            path = paths.WIKI_TRAINING_SPLIT_DIR + "%i.txt" % seq_len
            with open(path) as file:
                batch_sequences = []
                file.seek(start)
                for _ in range(batch_size):
                    line = file.readline()
                    if line == "":
                        break
                    sequence = line[:-1]
                    batch_sequences.append(sequence)
                file_positions[seq_len] = file.tell()
                yield Batch(batch_sequences, len(batch_sequences), seq_len)

    @staticmethod
    def benchmark_sequence_pairs(benchmark_name: str,
                                 split: str,
                                 seed: int=None,
                                 n: int=-1
                                 ) -> Iterator[Tuple[str, str]]:
        """Iterates over pairs of correct and corrupt sequences, stored in single-file format.

        Reads sequence lists from the Wikipedia output directory.

        :param benchmark_name: name of the benchmark
        :param split: Name of the partition. Either training, development or test.
        :param seed: seed used for shuffling sequence pairs
        :param n: number of sequence pairs
        :return: iterator over pairs (correct sequence, corrupt sequence)
        """
        correct_sequence_list = load_object(paths.WIKI_OUT_DIR + "%s_sequences.pkl" % split)
        correct_sequences = Wikipedia._read_sequences(paths.WIKI_SINGLE_DIR + "%s.txt" % split, correct_sequence_list,
                                                      seed=seed, n_sequences=n)
        corrupt_sequence_list = load_object(paths.WIKI_OUT_DIR + "%s_%s_sequences.pkl" % (benchmark_name, split))
        corrupt_sequences = Wikipedia._read_sequences(paths.WIKI_SINGLE_DIR+ "%s_%s.txt" % (benchmark_name, split),
                                                      corrupt_sequence_list, seed=seed, n_sequences=n)
        for correct, corrupt in zip(correct_sequences, corrupt_sequences):
            yield correct, corrupt
