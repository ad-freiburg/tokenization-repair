from numpy import cumsum

from project import src
from src.settings import paths
from src.sequence.transformation import pretokenize, split2words, get_word_positions, combine_mergable_tokens_all
from src.helper.pickle import load_object
from src.tokens.token_counter import is_word, get_count, sort_token_counters, most_frequent_tokens, \
    most_frequent_wiki_and_all_aspell_word_counts, sort_word_counters, sorted_word_counters_to_dict
from src.sequence.corruption import CorruptionType, Corruption
from src.sequence.splitting import get_best_splits
from src.tokens.aspell import is_aspell_word
from src.models.token_splitter import predict_split


def get_split_candidates(sequence, token_counters):
    candidates = [[sequence]]
    for i in range(len(sequence)):
        prefix = sequence[:i]
        if is_word(prefix, token_counters):
            follow_up_candidates = get_split_candidates(sequence[i:], token_counters)
            for follow_up in follow_up_candidates:
                candidates.append([prefix] + follow_up)
    return candidates


def sorted_token_counts(tokens, token_counters):
    return sorted([get_count(t, token_counters) for t in tokens])


def has_greater_counts(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] > b[i]:
            return True
        if a[i] < b[i]:
            return False
    return len(a) < len(b)


def positions_intersect(first, second):
    return second[0] < first[1]


def number_of_non_words(string, positions, word_counter):
    nonwords = 0
    for pos in positions:
        word = string[pos[0]:pos[1]]
        if (len(word) == 1 and word not in ["a", "A", "I"]) or word not in word_counter:
            nonwords += 1
    return nonwords


def candidates_from_word_positions(string, word_positions, word_counters, max_nonwords):
    candidates = [[]]
    nonwords = [0]
    for pos in word_positions:
        for c_i in range(len(candidates)):
            c = candidates[c_i]
            c_nonwords = nonwords[c_i]
            if len(c) == 0 or not positions_intersect(c[-1], pos):
                new_candidate = c.copy()
                new_nonwords = c_nonwords
                # fill potential gap:
                end_of_predecessor = 0 if len(c) == 0 else c[-1][1]
                if end_of_predecessor < pos[0]:
                    new_candidate.append((end_of_predecessor, pos[0]))
                    if string[end_of_predecessor:pos[0]] not in word_counters:
                        new_nonwords += 1
                # add to candidate:
                new_candidate.append(pos)
                # prune:
                if new_nonwords <= max_nonwords:
                    candidates.append(new_candidate)
                    nonwords.append(new_nonwords)
    for c in candidates:
        if len(c) == 0 or c[-1][1] < len(string):
            c.append((0 if len(c) == 0 else c[-1][1], len(string)))
    return candidates


def intersect_sorted_lists(a, b):
    intersect = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            intersect.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return intersect


def evaluate_candidate(string, candidate, word_dict, space_positions):
    candidate_space_positions = [pos[1] for pos in candidate[:-1]]
    intersect = intersect_sorted_lists(space_positions, candidate_space_positions)
    n_operations = len(candidate_space_positions) + len(space_positions) - 2 * len(intersect)
    n_nonwords = number_of_non_words(string, candidate, word_dict)
    score = n_operations + n_nonwords
    return score, n_operations, n_nonwords


def number_of_nonwords(tokens, word_counters, limit=None):
    nonwords = 0
    for token in tokens:
        if not is_word(token, word_counters):
            nonwords += 1
            if limit is not None and nonwords > limit:
                return nonwords
    return nonwords


def number_of_operations(positions, orig_space_positions):
    space_positions = [pos[1] for pos in positions[:-1]]
    intersect = intersect_sorted_lists(space_positions, orig_space_positions)
    num_ops = len(space_positions) + len(orig_space_positions) - 2 * len(intersect)
    return num_ops


def get_best_splits_naive(tokens, word_counters):
    #Finds the best splits based on number of nonwords and number of operations.
    orig_nonwords = sum([0 if is_word(token, word_counters) else 1 for token in tokens])
    orig_space_positions = cumsum([len(token) for token in tokens[:-1]])
    merged = "".join(tokens)
    word_positions = get_word_positions(merged, word_counters)
    candidates = candidates_from_word_positions(merged, word_positions, word_counters, orig_nonwords - 1)
    # original split as candidate:
    if len(tokens) == 1:
        original_split = [(0, len(merged))]
    else:
        original_split = [(0, orig_space_positions[0])]
        for i in range(len(orig_space_positions[:-1])):
            original_split.append((orig_space_positions[i], orig_space_positions[i + 1]))
        original_split.append((orig_space_positions[-1], len(merged)))
    candidates.append(original_split)
    candidates = set([tuple(c) for c in candidates])
    best_splits = []
    best_n_nonwords = len(merged)
    best_n_operations = len(merged)
    for c in candidates:
        c_words = [merged[pos[0]:pos[1]] for pos in c]
        c_n_nonwords = number_of_nonwords(c_words, word_counters, limit=best_n_nonwords)
        if c_n_nonwords <= best_n_nonwords:
            c_n_operations = number_of_operations(c, orig_space_positions)
        else:
            c_n_operations = len(merged)
        if c_n_nonwords < best_n_nonwords \
                or (c_n_nonwords == best_n_nonwords and c_n_operations < best_n_operations):
            best_n_nonwords = c_n_nonwords
            best_n_operations = c_n_operations
            best_splits = [c]
        elif c_n_nonwords == best_n_nonwords and c_n_operations == best_n_operations:
            best_splits.append(c)
    best_splits = [list(split) for split in best_splits]
    return best_splits


def get_best_split(tokens, word_counters):
    merged = "".join(tokens)
    best_words = []
    best_word_counts = []
    candidates = get_best_splits(tokens, word_counters)
    for i in range(len(candidates)):
        candidate = candidates[i]
        words = split2words(merged, candidate)
        word_counts = sorted_token_counts(words, word_counters)
        if i == 0 or has_greater_counts(word_counts, best_word_counts):
            best_words = words
            best_word_counts = word_counts
    return best_words


def get_relative_space_positions(sequence):
    split_sequence = sequence.split(' ')
    space_positions = cumsum([len(word) for word in split_sequence[:-1]])
    return space_positions


def get_predictions_from_space_positions(orignal_spaces, new_spaces):
    predictions = []
    orig_i = 0
    new_i = 0
    while orig_i < len(orignal_spaces) or new_i < len(new_spaces):
        # orig not in new: deletion (predict: insertion)
        if orig_i < len(orignal_spaces) and (new_i == len(new_spaces) or orignal_spaces[orig_i] < new_spaces[new_i]):
            insertion_pos = orignal_spaces[orig_i] + orig_i
            #print("space at %i was inserted" % insertion_pos)
            predictions.append(Corruption(CorruptionType.INSERTION, insertion_pos, ' '))
            orig_i += 1
        # new not in orig: insertion (predict: deletion)
        elif new_i < len(new_spaces) and (orig_i == len(orignal_spaces) or new_spaces[new_i] < orignal_spaces[orig_i]):
            deletion_pos = new_spaces[new_i] + orig_i
            #print("space at %i was deleted" % deletion_pos)
            predictions.append(Corruption(CorruptionType.DELETION, deletion_pos, ' '))
            new_i += 1
        # space conserved: no operation
        else:
            #orig_pos = orignal_spaces[orig_i] + orig_i
            #new_pos = new_spaces[new_i] + orig_i
            #print("space at %i == %i remains" % (orig_pos, new_pos))
            new_i += 1
            orig_i += 1
    return predictions


def get_predictions(original_sequence, predicted_sequence):
    original_spaces = get_relative_space_positions(original_sequence)
    predicted_spaces = get_relative_space_positions(predicted_sequence)
    predictions = get_predictions_from_space_positions(original_spaces, predicted_spaces)
    return predictions


def combine_mergable_tokens_until_two_correct(tokens, token_counters):
    token_lists = []
    t_i = 0
    while t_i < len(tokens):
        current = tokens[t_i]
        if not current[0]:
            token_lists.append((False, [current[1]]))
        else:
            words_to_merge = [current[1]]
            #original_space_positions = []
            consecutive_correct = 1 if is_word(current[1], token_counters) else 0
            while t_i < len(tokens) - 2:
                if tokens[t_i + 1][1] == ' ' and tokens[t_i + 2][0]:
                    word = tokens[t_i + 2][1]
                    if is_word(word, token_counters):
                        consecutive_correct += 1
                    else:
                        consecutive_correct = 0
                    if consecutive_correct == 2:
                        break
                    #original_space_positions.append(len(word)
                    #                                + (0 if len(original_space_positions) == 0
                    #                                   else original_space_positions[-1]))
                    words_to_merge.append(word)
                    t_i += 2
                else:
                    break
            token_lists.append((True, words_to_merge))
        t_i += 1
    return token_lists


def postprocess_maxmin(split, token_counters):
    # merge tokens
    merged_tokens = []
    for token in split:
        if len(merged_tokens) == 0:
            merged_tokens.append(token)
        else:
            last = merged_tokens[-1]
            last_count = get_count(last, token_counters)
            this_count = get_count(token, token_counters)
            merged = last + token
            merged_count = get_count(merged, token_counters)
            if merged_count > last_count or merged_count > this_count:
                merged_tokens[-1] = merged
            else:
                merged_tokens.append(token)
    # split tokens
    postprocessed = []
    for token in merged_tokens:
        best_count = get_count(token, token_counters)
        best_split_pos = None
        for split_pos in range(1, len(token)):
            prefix = token[:split_pos]
            suffix = token[split_pos:]
            smaller_count = min(get_count(prefix, token_counters), get_count(suffix, token_counters))
            print(prefix, suffix, smaller_count, best_count)
            if smaller_count > best_count:
                best_split_pos = split_pos
                best_count = smaller_count
        if best_split_pos is not None:
            postprocessed.append(token[:best_split_pos])
            postprocessed.append(token[best_split_pos:])
        else:
            postprocessed.append(token)
    return postprocessed


def postprocess_aspell(split, words):
    postprocessed = []
    made_changes = False
    for token in split:
        if is_aspell_word(token, words):
            postprocessed.append(token)
        else:
            solution_found = False
            for split_pos in range(1, len(token)):
                prefix = token[:split_pos]
                suffix = token[split_pos:]
                if is_aspell_word(prefix, words) and is_aspell_word(suffix, words):
                    postprocessed.append(prefix)
                    postprocessed.append(suffix)
                    solution_found = True
                    made_changes = True
                    break
            if not solution_found:
                postprocessed.append(token)
    if made_changes:
        print(split)
        print(postprocessed)
    return postprocessed


def postprocess_ml(split, token_counters, model):
    merged_tokens = []
    for token in split:
        if len(merged_tokens) == 0:
            merged_tokens.append(token)
        else:
            last = merged_tokens[-1]
            merged = last + token
            if is_word(merged, token_counters) and not predict_split(model, last, token, token_counters):
                merged_tokens[-1] = merged
            else:
                merged_tokens.append(token)
    postprocessed = []
    for token in merged_tokens:
        did_split = False
        for split_pos in range(1, len(token)):
            prefix = token[:split_pos]
            suffix = token[split_pos:]
            if is_word(prefix, token_counters) and is_word(suffix, token_counters) \
                    and predict_split(model, prefix, suffix, token_counters):
                postprocessed.append(prefix)
                postprocessed.append(suffix)
                did_split = True
                break
        if not did_split:
            postprocessed.append(token)
    return postprocessed


class DictionaryTokenizationCorrector:
    def __init__(self, n, use_aspell=True, postprocessing_method=None, verbose=False):
        if use_aspell:
            self.token_counters = most_frequent_wiki_and_all_aspell_word_counts(n)
        else:
            self.token_counters = most_frequent_tokens(n)
        self.postprocessing_method = postprocessing_method
        if postprocessing_method == "SVM":
            self.split_model = load_object(paths.SVM_SPLIT_MODEL)
        elif postprocessing_method == "RF":
            self.split_model = load_object(paths.RF_SPLIT_MODEL)
        self.verbose = verbose

    def predict(self, sequence):
        predicted = ""
        tokens = pretokenize(sequence)
        # token_lists = combine_mergable_tokens_until_two_correct(tokens, token_counters)
        token_lists = combine_mergable_tokens_all(tokens)
        for editable, token_list in token_lists:
            if editable:
                best_split = get_best_split(token_list, self.token_counters)
                if self.verbose and self.postprocessing_method is not None:
                    print(best_split)
                if self.postprocessing_method in ["SVM", "RF"]:
                    best_split = postprocess_ml(best_split, self.token_counters, self.split_model)
                elif self.postprocessing_method == "maxmin":
                    best_split = postprocess_maxmin(best_split, self.token_counters)
                predicted += " ".join(best_split)
            else:
                for token in token_list:
                    predicted += token
        predictions = get_predictions(sequence, predicted)
        dummy_probs = [1 for _ in predictions]
        return zip(dummy_probs, predictions), predicted


if __name__ == "__main__":
    """tc = most_frequent_tokens(100000)
    c1 = ["q", "u", "e", "s", "t", "i", "o", "n", "s"]
    c2 = ["questions"]
    v1 = sorted_token_counts(c1, tc)
    v2 = sorted_token_counts(c2, tc)
    print(c1, v1)
    print(c2, v2)
    print(has_greater_counts(c2, c1))"""

    # "thealgo rithm runsin" --> "the algorithm runs in"
    """original_spaces = [6, 11]
    new_spaces = [2, 11, 15]
    get_predictions(original_spaces, new_spaces)"""

    # best splits test:
    word_counters = most_frequent_wiki_and_all_aspell_word_counts(50000)
    while True:
        sequence = input("> ")
        tokens = sequence.split(' ')
        best_splits = get_best_splits(tokens, word_counters)
        merged = sequence.replace(' ', '')
        for split in best_splits:
            print(" ".join(split2words(merged, split)))

        #print()
        #print(" ".join(get_best_split(tokens, word_counters)))
