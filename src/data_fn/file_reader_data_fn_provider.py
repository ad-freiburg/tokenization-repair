from src.data_fn.wiki_data_fn_provider import WikiDataFnProvider
from src.helper.files import read_sequences


class FileReaderDataFnProvider(WikiDataFnProvider):
    def read_sequences(self):
        print("reading training data from %s" % str(self.dataset_file_path))
        return read_sequences(self.dataset_file_path)
