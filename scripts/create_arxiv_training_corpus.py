from project import src
from src.settings import paths, symbols
from src.helper.files import read_lines, write_lines
from src.arxiv.dataset import read_lines as read_training_lines
from src.helper.data_structures import select_most_frequent
from src.helper.pickle import dump_object

from nltk.tokenize import sent_tokenize


WRONG_SPLIT_ENDINGS = ("e.g.", " al.", "i.e.", "eq.", "Eq.", "Ref.", "Refs.", "Phys.", "Rev.", "Fig.", "I.", "II.",
                       "III.", "IV.", "V.", "op.")


def is_wrong_split(sentence):
    for ending in WRONG_SPLIT_ENDINGS:
        if sentence.endswith(ending):
            return True
    return False

def split_sentences(text):
    remerged = []
    for sentence in sent_tokenize(text):
        if len(remerged) == 0 or not is_wrong_split(sentence):
            remerged.append(sentence)
        else:
            remerged[-1] = remerged[-1] + sentence
    return remerged


if __name__ == "__main__":
    training_files = read_lines(paths.ARXIV_TRAINING_FILES)
    training_lines = []
    for file in training_files[1:]:
        lines = read_training_lines(paths.ARXIV_GROUND_TRUTH_DIR + file)
        training_lines += lines

    training_lines = [line for line in training_lines if line not in ("=", "[formula]", ".125in") and ".25in" not in line]

    print(len(training_lines), "lines")
    write_lines(paths.ARXIV_TRAINING_LINES, training_lines)

    print(sum(1 for line in training_lines if len(line) > 256), "length > 256")

    training_sentences = []
    for line in training_lines:
        sentences = split_sentences(line)
        training_sentences.extend(sentences)

    print(len(training_sentences), "sentences")
    write_lines(paths.ARXIV_TRAINING_SEQUENCES, training_sentences)

    char_frequencies = {}
    for sentence in training_sentences:
        for char in sentence:
            if char not in char_frequencies:
                char_frequencies[char] = 1
            else:
                char_frequencies[char] += 1
    encoder = {char: i for i, char in enumerate(sorted(select_most_frequent(char_frequencies, 200)))}
    encoder[symbols.SOS] = len(encoder)
    encoder[symbols.EOS] = len(encoder)
    encoder[symbols.UNKNOWN] = len(encoder)
    print(encoder)
    dump_object(encoder, paths.ARXIV_ENCODER_DICT)
