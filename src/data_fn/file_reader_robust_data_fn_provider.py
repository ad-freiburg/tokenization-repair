from src.data_fn.robust_data_fn_provicer import RobustDataFnProvider
from src.helper.files import read_sequences


class FileReaderRobustDataFnProvider(RobustDataFnProvider):
    def read_sequences(self):
        print("reading training data from %s" % str(self.training_file_path))
        return read_sequences(self.training_file_path)
