import numpy as np

from src.settings import symbols


def element_counters(list_of_sequences):
    counters = {}
    for sequence in list_of_sequences:
        for element in sequence:
            if element not in counters:
                counters[element] = 1
            else:
                counters[element] = counters[element] + 1
    return counters


def sort_counters(counters):
    return sorted([(counters[k], k) for k in counters], reverse=True)


def most_frequent_encoder_decoder_dicts(sequences, n=100, SOS=True):
    c = element_counters(sequences)
    c = sort_counters(c)
    c = c[:n]
    n = len(c)
    char2ix = {}
    ix2char = {}
    for i, pair in enumerate(c):
        char = pair[1]
        char2ix[char] = i
        ix2char[i] = char
    # unknown symbol, EOS and SOS
    char2ix[symbols.UNKNOWN] = n
    ix2char[n] = symbols.UNKNOWN
    char2ix[symbols.EOS] = n + 1
    ix2char[n + 1] = symbols.EOS
    if SOS:
        char2ix[symbols.SOS] = n + 2
        ix2char[n + 2] = symbols.SOS
    return char2ix, ix2char


def labels(sequence, encoder_dict):
    result = []
    if symbols.SOS in encoder_dict:
        result = [encoder_dict[symbols.SOS]]
    result += [encoder_dict[char] if char in encoder_dict else encoder_dict[symbols.UNKNOWN] for char in sequence]
    result += [encoder_dict[symbols.EOS]]
    return result


def encode(sequence, encoder_dict):
    seq_len = len(sequence)
    dimensions = len(encoder_dict)
    has_sos = symbols.SOS in encoder_dict
    unknown_symbol = symbols.UNKNOWN
    eos_symbol = symbols.EOS
    encoding = np.zeros((seq_len + (2 if has_sos else 1), dimensions), dtype=int)
    if has_sos:
        encoding[0, encoder_dict[symbols.SOS]] = 1
    for i, char in enumerate(sequence):
        ix = i +1 if has_sos else i
        if char in encoder_dict:
            encoding[ix, encoder_dict[char]] = 1
        else:
            encoding[ix, encoder_dict[unknown_symbol]] = 1
    encoding[-1, encoder_dict[eos_symbol]] = 1
    return encoding


def encode_batch(sequences, char2ix):
    n = len(sequences)
    encoding_len = len(sequences[0]) + 2
    dims = len(char2ix)

    X = np.zeros((n, encoding_len, dims), np.int8)
    y_labels = np.zeros((n, encoding_len), np.int8)
    for s_ix, sequence in enumerate(sequences):
        X[s_ix, :, :] = encode(sequence, char2ix)
        y_labels[s_ix, :] = labels(sequence, char2ix)

    return X, y_labels


def decode(sequence, decoder_dict):
    indices = np.argmax(sequence, axis=1)
    chars = [decoder_dict[i] for i in indices]
    return "".join(chars)


def predictions(probabilities):
    return np.argmax(probabilities, axis=1)


def predicted_chars(probabilities, ix2char):
    return [ix2char[i] for i in predictions(probabilities)]
