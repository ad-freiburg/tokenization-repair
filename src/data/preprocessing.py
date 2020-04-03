def preprocess_sequence(sequence):
    sequence = sequence.replace('\xa0', ' ')
    sequence = sequence.strip()
    while "  " in sequence:
        sequence = sequence.replace("  ", ' ')
    return sequence