from src.data_fn.robust_data_fn_provicer import RobustDataFnProvider
from src.helper.files import read_sequences
from src.settings import paths


class ArxivRobustDataFnProvider(RobustDataFnProvider):
    def read_sequences(self):
        return read_sequences(paths.ARXIV_TRAINING_SEQUENCES)
