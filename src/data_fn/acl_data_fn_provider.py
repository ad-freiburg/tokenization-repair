from src.data_fn.wiki_data_fn_provider import WikiDataFnProvider
from src.helper.files import read_sequences
from src.settings import paths


class ACLDataFnProvider(WikiDataFnProvider):
    def read_sequences(self):
        return read_sequences(paths.ACL_TRAINING_FILE)
