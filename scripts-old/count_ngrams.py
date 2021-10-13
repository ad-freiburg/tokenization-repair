import sys

import project
from src.datasets.wikipedia import Wikipedia
from src.interactive.sequence_generator import interactive_sequence_generator
from src.ngram.tokenizer import Tokenizer, Token
from src.ngram.unigram_holder import UnigramHolder
from src.settings import paths
from src.helper.pickle import dump_object
from src.helper.time import time_diff, timestamp
from src.helper.data_structures import sort_dict_by_value
from src.ngram.bigram_holder import BigramHolder


K = int(1e3)
K10 = int(1e4)
K100 = int(1e5)
M = int(1e6)


def count_unigrams(n_sequences: int):
    total_start = timestamp()

    tokenizer = Tokenizer()
    counts_delim = {}
    counts_no_delim = {}

    tokenization_time = 0

    for s_i, sequence in enumerate(Wikipedia.training_sequences(n_sequences)):
        start = timestamp()
        tokens = tokenizer.tokenize(sequence)
        tokens[0].delimiter_before = True
        tokenization_time += time_diff(start)
        for token in tokens:
            counts = counts_delim if token.delimiter_before else counts_no_delim
            if token.text not in counts:
                counts[token.text] = 1
            else:
                counts[token.text] += 1
        if (s_i + 1) % K10 == 0:
            print("%ik sequences, %.2f s total time, %.2f s tokenization" % ((s_i + 1) / K,
                                                                             time_diff(total_start),
                                                                             tokenization_time))
        if (s_i + 1) % M == 0:
            print("saving...")
            dump_object(counts_delim, paths.UNIGRAM_DELIM_FREQUENCY_DICT)
            dump_object(counts_no_delim, paths.UNIGRAM_NO_DELIM_FREQUENCY_DICT)


def print_top_unigrams():
    holder = UnigramHolder()
    sorted_unigrams = sort_dict_by_value(holder.frequencies)
    for unigram in sorted_unigrams[:100]:
        print(unigram)


def search_tokens():
    n = int(sys.argv[2])
    tokenizer = Tokenizer()
    for query in interactive_sequence_generator():
        if query.startswith(' '):
            query_token = Token(query[1:], True)
        else:
            query_token = Token(query, False)
        for sequence in Wikipedia.training_sequences(n):
            tokens = tokenizer.tokenize(sequence)
            if query_token in tokens:
                print(sequence)


def query_unigrams():
    holder = UnigramHolder()
    for query in interactive_sequence_generator():
        print(holder.get(query))


def count_bigrams(n_sequences: int):
    tokenizer = Tokenizer()
    holder = BigramHolder()
    for s_i, sequence in enumerate(Wikipedia.training_sequences(n_sequences)):
        tokens = tokenizer.tokenize(sequence)
        texts = [token.text for token in tokens]
        for i in range(len(tokens) - 1):
            bigram = texts[i:(i + 2)]
            holder.increment(bigram)
        if (s_i + 1) % K10 == 0:
            print("%ik sequences" % ((s_i + 1) / K))
        if (s_i + 1) % M == 0:
            print("saving...")
            holder.save()
    holder.save()


def print_top_bigrams():
    holder = BigramHolder.load()
    sorted = sort_dict_by_value(holder.bigram_counts)
    for bigram, frequency in sorted[:100]:
        print(holder.decode(bigram), frequency)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "count":
        n_sequences = int(sys.argv[2])
        count_unigrams(n_sequences)
    elif mode == "print":
        print_top_unigrams()
    elif mode == "search":
        search_tokens()
    elif mode == "query":
        query_unigrams()
    elif mode == "count-bigrams":
        n_sequences = int(sys.argv[2])
        count_bigrams(n_sequences)
    elif mode == "top-bigrams":
        print_top_bigrams()
