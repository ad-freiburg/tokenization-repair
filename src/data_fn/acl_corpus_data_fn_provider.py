import random

from src.data_fn.wiki_data_fn_provider import WikiDataFnProvider
from src.helper.files import read_sequences
from src.settings import paths


class ACLCorpusDataFnProvider(WikiDataFnProvider):
    def read_sequences(self):
        sequences = list(read_sequences(paths.ACL_CORPUS_TRAINING_FILE))
        random.shuffle(sequences)
        for sequence in sequences:
            yield sequence
