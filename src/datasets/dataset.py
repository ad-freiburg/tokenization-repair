from enum import Enum
from src.settings import paths


class DatasetName(Enum):
    EUROPARL = 0
    WIKIPEDIA = 1


class DatasetPartition(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


def get_file_path(dataset_name, dataset_partition):
    if dataset_name == DatasetName.WIKIPEDIA and dataset_partition == DatasetPartition.TRAINING:
        return paths.WIKI_TRAINING_FILE
    elif dataset_name == DatasetName.WIKIPEDIA and dataset_partition == DatasetPartition.VALIDATION:
        return paths.WIKI_VALIDATION_FILE
    elif dataset_name == DatasetName.WIKIPEDIA and dataset_partition == DatasetPartition.TEST:
        return paths.WIKI_TEST_FILE
    elif dataset_name == DatasetName.EUROPARL and dataset_partition == DatasetPartition.TRAINING:
        return paths.EUROPARL_TRAINING_FILE
    elif dataset_name == DatasetName.EUROPARL and dataset_partition == DatasetPartition.VALIDATION:
        return paths.EUROPARL_VALIDATION_FILE
    elif dataset_name == DatasetName.EUROPARL and dataset_partition == DatasetPartition.TEST:
        return paths.EUROPARL_TEST_FILE
    raise NotImplementedError("Unknown dataset %s or partition %s." % (str(dataset_name), str(dataset_partition)))


def yield_first_lines(dataset_name, dataset_partition, n):
    path = get_file_path(dataset_name, dataset_partition)
    with open(path, encoding="UTF8") as file:
        for i, line in enumerate(file):
            if n is not None and i == n:
                break
            yield line[:-1]


class Dataset:
    def __init__(self, dataset_name):
        self.name = dataset_name

    def get_training_sequences(self, n=None):
        return yield_first_lines(self.name, DatasetPartition.TRAINING, n)

    def get_validation_sequences(self, n=None):
        return yield_first_lines(self.name, DatasetPartition.VALIDATION, n)

    def get_test_sequences(self, n=None):
        return yield_first_lines(self.name, DatasetPartition.TEST, n)
