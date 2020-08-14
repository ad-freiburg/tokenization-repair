from src.data_fn.robust_data_fn_provicer import RobustDataFnProvider
from src.helper.files import read_sequences
from src.settings import paths


class ACLRobustDataFnProvider(RobustDataFnProvider):
    def read_sequences(self):
        return read_sequences(paths.ACL_TRAINING_FILE)
