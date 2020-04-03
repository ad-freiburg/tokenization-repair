from src.sequence.corruption import CorruptionType, Corruption
from src.sequence.transformation import is_number, is_letter, is_uppercase, is_lowercase


class TokenizationRule:
    @staticmethod
    def predict(chars_before, char, chars_after):
        raise NotImplementedError


def delete_space_before_char(del_before, char, chars_after):
    if char == ' ' and len(chars_after) > 0 and chars_after[0] == del_before:
        return CorruptionType.INSERTION
    return None


class DeleteSpaceBeforeComma(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        return delete_space_before_char(',', char, chars_after)


class DeleteSpaceBeforePoint(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        if chars_before[-2:] == "\xa0\xa0":
            return None
        tmp = delete_space_before_char('.', char, chars_after)
        return tmp


class DeleteSpaceBeforeClosingBracket(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        return delete_space_before_char(')', char, chars_after)


class DeleteSpaceBeforeSpecialCharacters(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        if len(chars_after) == 0:
            return None
        if char == ' ' and chars_after[0] in ['?']:
            return CorruptionType.INSERTION
        return None


class InsertSpaceAfterComma(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        char_before = '' if len(chars_before) == 0 else chars_before[-1]
        char_after = '' if len(chars_after) == 0 else chars_after[0]
        if char != ' ' and char_before == ',' and not is_number(char_after):
            return CorruptionType.DELETION
        return None


class DeleteSpaceBetweenNumbers(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        # exception before '000'
        if chars_after[:3] == "000":
            return None
        # space between numbers
        if char == ' ' \
                and len(chars_before) > 0 and is_number(chars_before[-1]) \
                and len(chars_after) > 0 and is_number(chars_after[0]):
            return CorruptionType.INSERTION
        # # .#
        if char == ' ' \
                and len(chars_before) > 0 and is_number(chars_before[-1]) \
                and len(chars_after) > 1 and chars_after[0] in [',', '.'] and is_number(chars_after[1]):
            return CorruptionType.INSERTION
        """"# #. #
        if char == ' ' \
                and len(chars_before) > 1 and is_number(chars_before[-2]) and chars_before[-1] in [',', '.'] \
                and len(chars_after) > 0 and is_number(chars_after[0]):
            return CorruptionType.INSERTION"""
        return None


class InsertSpaceBetweenLetterAndNumber(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        # exceptions:
        if char + chars_after[:2] in ['st ', 'nd', 'rd '] or char == 's':
            return None
        if char != ' ' and len(chars_before) > 0 \
            and ((is_number(chars_before[-1]) and is_letter(char))
                 or (is_letter(chars_before[-1]) and is_number(char))):
            return CorruptionType.DELETION
        return None


class InsertSpaceBeforeOpeningBracket(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        if char == '(' and len(chars_before) > 0 and chars_before[-1:] != ' ':
            return CorruptionType.DELETION
        return None


class InsertSpaceBetweenCamelCase(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        if is_lowercase(chars_before[-1:]) and is_uppercase(char):
            return CorruptionType.DELETION
        return None


class InsertSpaceAfterApostrophS(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        if char != ' ' and is_letter(chars_before[-3:-2]) and chars_before[-2:] == "'s":
            return CorruptionType.DELETION
        return None


class InsertSpaceAfterSpecialCharacters(TokenizationRule):
    @staticmethod
    def predict(chars_before, char, chars_after):
        if char != ' ' and len(chars_before) > 0 and chars_before[-1] in [';']:
            return CorruptionType.DELETION
        return None


class RuleBasedTokenizer:
    def __init__(self):
        self.rules = [DeleteSpaceBeforeComma,
                      DeleteSpaceBeforePoint,
                      DeleteSpaceBeforeClosingBracket,
                      DeleteSpaceBetweenNumbers,
                      InsertSpaceAfterComma,
                      InsertSpaceBeforeOpeningBracket,
                      InsertSpaceAfterSpecialCharacters]

    def predict(self, sequence):
        predictions = []
        for pos in range(len(sequence)):
            deletion = False
            insertion = False
            before = sequence[max(0, pos - 3):pos]
            char = sequence[pos]
            after = sequence[(pos + 1):(pos + 4)]
            for rule in self.rules:
                prediction_type = rule.predict(before, char, after)
                if prediction_type == CorruptionType.INSERTION:
                    insertion = True
                elif prediction_type == CorruptionType.DELETION:
                    deletion = True
            if insertion:
                predictions.append(Corruption(CorruptionType.INSERTION, pos, ' '))
            if deletion:
                predictions.append(Corruption(CorruptionType.DELETION, pos, ' '))
        fake_probabilities = [1 for _ in range(len(predictions))]
        return zip(fake_probabilities, predictions)



do_print = False


def verbose(*args):
    if do_print:
        print(args)


if __name__ == "__main__":
    from sequence_corruptor import SequenceCorruptor, Corruption
    from dataset import Dataset, DatasetNames

    rulor = RuleBasedTokenizer()

    #dataset = Dataset(DatasetNames.WIKIPEDIA)
    dataset = Dataset(DatasetNames.EUROPARL_EN)

    n_sequences = 10000

    n = None  # 1
    p = 0.05
    sequences = dataset.get_training_sequences(n_sequences)
    char2ix, ix2char = dataset.get_character_dictionaries()

    corruptor = SequenceCorruptor(tokenization=True, n=n, p=p, seed=42, char_dict=ix2char)

    true_positives = 0
    false_positives = 0
    n_insertions = 0
    n_deletions = 0

    for sequence in sequences:
        corruptions, corrupted = corruptor.corrupt(sequence)
        predictions = rulor.predict(corrupted)
        for prediction in predictions:
            prediction = prediction[1]
            pos = prediction.position
            correct = prediction in corruptions
            verbose(corrupted[max(0, pos - 20):pos], prediction.type, corrupted[pos:(pos + 20)],
                  "CORRECT" if correct else "WRONG")
            if correct:
                true_positives += 1
            else:
                false_positives += 1

    print("%i true positives" % true_positives)
    print("%i false positives" % false_positives)
