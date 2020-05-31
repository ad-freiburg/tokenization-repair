def get_inserted_position(original, misspelled):
    if len(original) >= len(misspelled):
        return None
    for i, m in enumerate(misspelled):
        if i == len(original) or m != original[i]:
            return i
    assert False


def get_inserted_nonspace_positions(original, misspelled):
    inserted_positions = set()
    pos = 0
    for orig, missp in zip(original.split(), misspelled.split()):
        inserted = get_inserted_position(orig, missp)
        if inserted is not None:
            inserted_positions.add(pos + inserted)
        pos += len(missp)
    return inserted_positions


def remove_inserted_nonspace_characters(sequence, nonspace_insertions):
    processed = ""
    nonspace_i = 0
    for i, char in enumerate(sequence):
        if (char == ' ' and not processed.endswith(' ')) or (char != ' ' and nonspace_i not in nonspace_insertions):
            processed += char
        if char != ' ':
            nonspace_i += 1
    return processed


def tolerant_preprocess_sequences(original, correct, corrupt, predicted):
    insertions = get_inserted_nonspace_positions(original, correct)
    correct = remove_inserted_nonspace_characters(correct, insertions)
    corrupt = remove_inserted_nonspace_characters(corrupt, insertions)
    predicted = remove_inserted_nonspace_characters(predicted, insertions)
    return correct, corrupt, predicted


def is_correct_tolerant(original, correct, corrupt, predicted):
    correct, corrupt, predicted = tolerant_preprocess_sequences(original, correct, corrupt, predicted)
    return predicted == correct


if __name__ == "__main__":
    print(is_correct_tolerant("Hello world.",
                              "Hellox world.",
                              "Helloxworld.",
                              "Hello xworld."))
    print(is_correct_tolerant("""It was founded in November 1991 and led by Nina Andreyeva, a university teacher who was well known for her 1988 letter "I cannot give up my principles".""",
                              """Izt was founded in November 1991 and sled by Nina Andreyeva, a univeristy teacher who was well known for her 1988 letter "Ic cannot give up my principles".""",
                              """Izt was founded in No vember 1991 and sled by Nina Andreyeva, auniveristy teacher who was well known for her 1988 letter "Ic cannot give up my principles".""",
                              """Izt was founded in November 1991 and s led by Nina Andreyeva, a univeristy teacher who was well known for her 1988 letter "Ic cannot give up my principles"."""))
    print(is_correct_tolerant("The cat eats fish.",
                              "The cat eatz fish.",
                              "The cat eatz fish.",
                              "The cat eat z fish."))
