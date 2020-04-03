from src.helper.pickle import load_object, dump_object
from src.settings import paths
from src.corrector.threshold_fitter import ThresholdFitter


class ThresholdFitterHolder:
    def __init__(self,
                 model_name: str,
                 fwd_model_name: str,
                 bwd_model_name: str,
                 benchmark_name: str,
                 insert: bool):
        self.model_name = model_name
        self.fwd_model_name = fwd_model_name
        self.bwd_model_name = bwd_model_name
        self.benchmark_name = benchmark_name
        self.insert = insert

    def _file_name(self):
        return "%s%s%s?%s?%s.txt" % (
            self.model_name if self.model_name is not None else "",
            self.fwd_model_name if self.fwd_model_name is not None else "",
            ("?" + self.bwd_model_name) if self.bwd_model_name is not None else "",
            self.benchmark_name,
            "insert" if self.insert else "delete"
        )

    def _file(self):
        return paths.THRESHOLD_FITTER_DIR + self._file_name()

    def load(self):
        return load_object(self._file())

    def dump(self, fitter: ThresholdFitter):
        dump_object(fitter, self._file())
