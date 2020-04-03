from src.settings import paths


def get_aspell_words():
    with open(paths.ASPELL_WORD_FILE) as file:
        words = file.readlines()
    words = [w[:-1] for w in words]
    words = set([w for w in words if len(w) > 1 or w in ["A", "a", "I"]])
    return words


def is_aspell_word(word, aspell_words):
    return word in aspell_words or word.lower() in aspell_words
