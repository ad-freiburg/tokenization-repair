import abc

from src.models.char_lm.character_language_model import CharacterLanguageModel
from src.helper.characters import is_space
from src.helper.data_structures import gather


class TokenCorrector:
    def __init__(self,
                 model: CharacterLanguageModel,
                 insert=True,
                 delete=True):
        self.model = model
        self.insert = insert
        self.delete = delete
        self._space_index = self.model.get_encoder().encode_char(' ')

    @abc.abstractmethod
    def correct(self, sequence):
        raise NotImplementedError()

    def _get_probabilities(self, sequence):
        result = self.model.predict(sequence)
        sequence_probabilities = gather(result["probabilities"], result["labels"][1:-1])
        blank_insertion_probabilities = result["insertion_probabilities"][:, self._space_index]
        blank_deletion_probabilities = [0 if not is_space(char) else p
                                        for char, p in zip(sequence, result["deletion_probabilities"])]
        return {"sequence": sequence_probabilities,
                "insertion": blank_insertion_probabilities,
                "deletion": blank_deletion_probabilities}
