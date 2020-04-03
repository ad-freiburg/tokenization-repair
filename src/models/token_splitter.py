import numpy as np

from src.tokens.token_counter import get_count, is_word
from src.sequence.transformation import pretokenize, combine_mergable_tokens_all


def feature_matrix(prefix_counts, suffix_counts, merged_counts):
    X = np.zeros((len(merged_counts), 3), dtype=int)
    X[:, 0] = prefix_counts
    X[:, 1] = suffix_counts
    X[:, 2] = merged_counts
    return X


def sequences2mergable_token_lists(sequences):
    mergable_token_lists = []
    for sequence in sequences:
        tokens = pretokenize(sequence)
        token_lists = combine_mergable_tokens_all(tokens)
        mergable_token_lists += [tl[1] for tl in token_lists if tl[0]]
    return mergable_token_lists


def negative_examples(token_lists, word_counters):
    neg_c_tok = []
    neg_c_pre = []
    neg_c_suf = []
    for token_list in token_lists:
        for token in token_list:
            for split_pos in range(1, len(token)):
                prefix = token[:split_pos]
                suffix = token[split_pos:]
                if is_word(prefix, word_counters) and is_word(suffix, word_counters):
                    token_count = get_count(token, word_counters)
                    prefix_count = get_count(prefix, word_counters)
                    suffix_count = get_count(suffix, word_counters)
                    neg_c_tok.append(token_count)
                    neg_c_pre.append(prefix_count)
                    neg_c_suf.append(suffix_count)
    X = feature_matrix(neg_c_pre, neg_c_suf, neg_c_tok)
    y = [0] * len(neg_c_tok)
    return X, y


def positive_examples(token_lists, word_counters):
    pos_c_tok = []
    pos_c_pre = []
    pos_c_suf = []
    for token_list in token_lists:
        for prefix, suffix in zip(token_list[:-1], token_list[1:]):
            merged = prefix + suffix
            if is_word(merged, word_counters):
                merged_count = get_count(merged, word_counters)
                prefix_count = get_count(prefix, word_counters)
                suffix_count = get_count(suffix, word_counters)
                pos_c_tok.append(merged_count)
                pos_c_pre.append(prefix_count)
                pos_c_suf.append(suffix_count)
    X = feature_matrix(pos_c_pre, pos_c_suf, pos_c_tok)
    y = [1] * len(pos_c_tok)
    return X, y


def get_X_y(sequences, word_counters):
    token_lists = sequences2mergable_token_lists(sequences)
    X_positive, y_positive = positive_examples(token_lists, word_counters)
    X_negative, y_negative = negative_examples(token_lists, word_counters)
    X = np.concatenate((X_positive, X_negative), axis=0)
    y = np.array(y_positive + y_negative)
    return X, y


def predict_split(model, prefix, suffix, word_counters):
    merged = prefix + suffix
    prefix_count = get_count(prefix, word_counters)
    suffix_count = get_count(suffix, word_counters)
    merged_count = get_count(merged, word_counters)
    X = np.array([prefix_count, suffix_count, merged_count]).reshape((1, -1))
    y_hat = model.predict(X)
    return y_hat[0] == 1
