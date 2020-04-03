from project import src
from src.helper.files import file_exists
from src.helper.pickle import load_object, dump_object
from src.settings import paths


def sort_token_counters(word_counters):
    """
    Sorts the given counters.

    :param word_counters: Dictionary with counts for all tokens.
    :return: List of (count, token) pairs sorted by counts from greatest to smallest.
    """
    return sorted([(word_counters[w], w) for w in word_counters], reverse=True)


def get_count(token, token_counters):
    if len(token) == 1 and token not in ['a', 'A', 'I'] and not token.isnumeric():
        return 0
    return 0 if token not in token_counters else token_counters[token]


def is_word(word, word_counters):
    return get_count(word, word_counters) > 0


def sort_word_counters(word_counters):
    sorted_pairs = sorted([(word_counters[w], w) for w in word_counters], reverse=True)
    return sorted_pairs


def sorted_word_counters_to_dict(sorted_pairs):
    return {p[1]: p[0] for p in sorted_pairs}


def most_frequent(word_counters, k):
    sorted_pairs = sort_word_counters(word_counters)
    sorted_pairs = sorted_pairs[:k]
    frequent = sorted_word_counters_to_dict(sorted_pairs[:k])
    return frequent


def most_frequent_tokens(n):
    if n is None:
        print("loading counters...")
        return load_object(paths.WIKI_TOKEN_COUNTERS)
    most_frequent_path = paths.WIKI_MOST_FREQUENT_TOKENS % n
    if file_exists(most_frequent_path):
        print("loading most frequent counters...")
        return load_object(most_frequent_path)
    sorted_token_counters_path = paths.WIKI_SORTED_TOKEN_COUNTERS
    if file_exists(sorted_token_counters_path):
        print("loading sorted counters...")
        sorted_token_counters = load_object(sorted_token_counters_path)
    else:
        print("loading counters...")
        token_counters = load_object(paths.WIKI_TOKEN_COUNTERS)
        print("sorting counters...")
        sorted_token_counters = sort_word_counters(token_counters)
        pickle_dump(sorted_token_counters, sorted_token_counters_path)
    most_frequent = sorted_word_counters_to_dict(sorted_token_counters[:n])
    if not file_exists(most_frequent_path):
        pickle_dump(most_frequent, most_frequent_path)
    return most_frequent


def most_frequent_wiki_and_all_aspell_word_counts(k_most_frequent):
    path = paths.WIKI_AND_ASPELL_TOKEN_COUNTERS % k_most_frequent
    if file_exists(path):
        return load_object(path)
    words = most_frequent_tokens(k_most_frequent)
    wiki_word_counters = load_object(paths.WIKI_TOKEN_COUNTERS)
    with open(paths.ASPELL_WORD_FILE) as f:
        for line in f:
            word = line[:-1]
            if word not in words:
                words[word] = wiki_word_counters[word] if word in wiki_word_counters else 0
    dump_object(words, path)
    return words
