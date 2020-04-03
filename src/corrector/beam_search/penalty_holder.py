from typing import Tuple

from src.helper.files import file_exists
from src.helper.pickle import load_object, dump_object
from src.settings import paths


class PenaltyHolder:
    def __init__(self, autosave: bool = True):
        if file_exists(paths.BEAM_SEARCH_PENALTY_FILE):
            self.penalties = load_object(paths.BEAM_SEARCH_PENALTY_FILE)
        else:
            self.penalties = {}
        self.autosave = autosave

    def save(self):
        dump_object(self.penalties, paths.BEAM_SEARCH_PENALTY_FILE)

    @staticmethod
    def _key(model_name: str, benchmark_name: str):
        return model_name, benchmark_name

    def set(self,
            model_name: str,
            benchmark_name: str,
            insertion_penalty: float,
            deletion_penalty: float):
        key = self._key(model_name, benchmark_name)
        self.penalties[key] = (insertion_penalty, deletion_penalty)
        if self.autosave:
            self.save()

    def get(self,
            model_name: str,
            benchmark_name: str) -> Tuple[float, float]:
        key = self._key(model_name, benchmark_name)
        return self.penalties[key]
