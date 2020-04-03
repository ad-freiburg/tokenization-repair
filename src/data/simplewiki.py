import random

from src.settings import folds, paths, file_names
from src.helper.pickle import load_object
from src.helper.files import read_file


class Simplewiki:
    @staticmethod
    def get_evaluation_files(select_folds):
        files = set()
        if 1 in select_folds:
            files |= set(folds.FOLD2) & set(folds.FOLD3)
        if 2 in select_folds:
            files |= set(folds.FOLD1) & set(folds.FOLD3)
        if 3 in select_folds:
            files |= set(folds.FOLD1) & set(folds.FOLD2)
        files = [file[len(file_names.SIMPLEWIKI_FILE_PREFIX):] for file in files]
        return sorted(files)

    @staticmethod
    def get_evaluation_files_shuffled(n=None, select_folds=[1, 2, 3], seed=42):
        files = Simplewiki.get_evaluation_files(select_folds=select_folds)
        rdm = random.Random()
        rdm.seed(seed)
        rdm.shuffle(files)
        if n is not None:
            return files[:n]
        else:
            return files

    @staticmethod
    def get_evaluation_samples(file_name):
        file_name = paths.WIKI_EVALUATION_DIR + file_name
        original_lines = read_file(file_name + file_names.ORIGINAL_SUFFIX).split('\n')
        original_lines = [line for line in original_lines if len(line) > 0]
        corrupt_lines = read_file(file_name + file_names.CORRUPT_SUFFIX).split('\n')
        corrupt_lines = [line for line in corrupt_lines if len(line) > 0]
        corruptions = load_object(file_name + file_names.CORRUPTIONS_SUFFIX)
        assert(len(corrupt_lines) == len(original_lines))
        assert(len(corruptions) == len(original_lines))
        return list(zip(original_lines, corrupt_lines, corruptions))

    @staticmethod
    def get_dictionaries():
        encoder = load_object(paths.WIKI_ENCODER_DICT)
        decoder = load_object(paths.WIKI_DECODER_DICT)
        return encoder, decoder


if __name__ == "__main__":
    files = Simplewiki.get_evaluation_files()
    for f in files:
        print()
        print(f)
        samples = Simplewiki.get_evaluation_samples(f)
        for original, corrupt, corruptions in samples:
            print(original)
            print(corrupt)
            print(corruptions)
