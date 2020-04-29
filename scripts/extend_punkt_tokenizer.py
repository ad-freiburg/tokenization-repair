from typing import Tuple

import sys

import project
from src.helper.files import read_sequences
from src.settings import paths
from src.helper.pickle import load_object, dump_object


def sequences():
    N = int(sys.argv[-1])
    for s_i, sequence in enumerate(read_sequences(paths.WIKI_TRAINING_PARAGRAPHS)):
        if s_i == N:
            break
        elif (s_i + 1) % 100000 == 0:
            print("%.1fM sequences" % ((s_i + 1) / int(1e6)))
        yield sequence


def contains_alphabetic(text: str):
    for char in text:
        if char.isalpha():
            return True
    return False


def is_excluded_abbreviation(candidate: str) -> bool:
    return candidate.startswith("km") or candidate.startswith("sq.")


def abbreviation_candidate(token: str) -> Tuple[bool, str]:
    candidate = token[:-1]
    is_candidate = False
    if len(token) > 1 and token[-1] == '.':
        n_points = sum([1 if c == '.' else 0 for c in candidate]) + 1
        if len(token) / n_points <= 5:
            if candidate[-1] != ')' and '"' not in candidate and contains_alphabetic(candidate) \
                    and not candidate.endswith("..") and not is_excluded_abbreviation(candidate):
                is_candidate = True
    return is_candidate, candidate


if __name__ == "__main__":
    MODE = sys.argv[1]

    CANDIDATE_FILE = paths.DUMP_DIR + "tmp_punkt_candidates.pkl"
    COUNT_FILE = paths.DUMP_DIR + "tmp_punkt_counts.pkl"
    ABBREVIATIONS_FILE = paths.EXTENDED_PUNKT_ABBREVIATIONS

    if MODE == "candidates":
        all_tokens = set()
        for sequence in sequences():
            for token in sequence.split():
                all_tokens.add(token)
        candidates = set()
        for token in all_tokens:
            is_candidate, candidate = abbreviation_candidate(token)
            if is_candidate:
                candidates.add(candidate)
        dump_object(candidates, CANDIDATE_FILE)

    elif MODE == "print-candidates":
        candidates = load_object(CANDIDATE_FILE)
        for c in sorted(candidates):
            print(c)

    elif MODE == "count":
        counts = {candidate: [0, 0] for candidate in load_object(CANDIDATE_FILE)}
        for sequence in sequences():
            tokens = sequence.split()
            for token in tokens:
                if token in counts:
                    counts[token][0] += 1
                if token[-1] == '.':
                    candidate = token[:-1]
                    if candidate in counts:
                        counts[candidate][1] += 1
        dump_object(counts, COUNT_FILE)

    elif MODE == "print":
        print_all = "all" in sys.argv
        counts = load_object(COUNT_FILE)
        abbreviations = set()
        for candidate in sorted(counts):
            is_candidate, _ = abbreviation_candidate(candidate + '.')
            if is_candidate:
                no_pt, pt = counts[candidate]
                if print_all or (pt >= 100 and (no_pt == 0 or pt / no_pt > 2)):
                    print(candidate, no_pt, pt)
                    abbreviations.add(candidate)
        if not print_all:
            dump_object(abbreviations, ABBREVIATIONS_FILE)
