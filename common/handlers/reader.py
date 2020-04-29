import os

import numpy as np

from constants import encode, decode, encode_operation
from utils.utils import (
    extract_file_name, read_all, context_generator, edits_generator,
    read_shuffled_lines)
from utils.edit_operations import detailed_edit_operations
from utils.logger import logger


class Reader:
    def __init__(self, config):
        self.config = config

    def read_test_pairs(self):
        logger.log_debug(self.config.test_correct_path)
        with open(self.config.test_correct_path, 'r') as correct_fil,\
                open(self.config.test_corrupt_path, 'r') as corrupt_fil:
            data = sorted(list(enumerate(zip(correct_fil, corrupt_fil))),
                          key=(lambda x: len(x[1][0])),
                          reverse=True)
            for idx, pair in data:
                pair = pair[0][:-1], pair[1][:-1]
                yield ('line_num_%.6d.txt' % idx,) + pair
        return

    def read_valid_pairs(self):
        logger.log_debug(self.config.valid_correct_path)
        with open(self.config.valid_correct_path, 'r') as correct_fil,\
                open(self.config.valid_corrupt_path, 'r') as corrupt_fil:
            data = sorted(list(enumerate(zip(correct_fil, corrupt_fil))),
                          key=(lambda x: len(x[1][0])),
                          reverse=True)
            for idx, pair in data:
                pair = pair[0][:-1], pair[1][:-1]
                yield pair
        return

    def read_train_lines(self):
        with open(self.config.train_path, 'r') as train_fil:
            lines = sum(1 for ln in train_fil)
        return lines, self.read_file(self.config.train_path)

    def read_file(self, path):
        with open(path, 'r') as fil:
            for line in fil:
                yield line[:-1]
        return

    """
    def read_valid_triples(self):
        return self.edits_generator(self.read_valid_pairs(), self.config.history_length)

    def edits_generator(self, pairs_gen, context_length):
        for correct, corrupt in pairs_gen:
            editops = detailed_edit_operations(corrupt, correct)
            correct_data = encode(correct)
            corrupt_data = encode(corrupt)

            for idx_corrupt, idx_correct, action in editops:
                before = correct_data[max(0, idx_correct - context_length): idx_correct]
                after = corrupt_data[idx_corrupt + 1: idx_corrupt + context_length + 1]
                current = corrupt_data[idx_corrupt: idx_corrupt + 1]
                action_code = np.array([encode_operation(action)])

                yield before, current, after, action_code
        return
    """
