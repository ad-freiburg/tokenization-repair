from termcolor import colored


def remove_inserted_char(misspelled: str, original: str) -> str:
    if misspelled == original or len(misspelled) < len(original):
        return misspelled
    elif len(misspelled) > len(original):
        return original
    for i in range(len(misspelled)):
        if misspelled[i] != original[i]:
            return misspelled[:i] + misspelled[(i + 1):]


def remove_inserted_chars(misspelled: str, original: str) -> str:
    misspelled_tokens = misspelled.split(' ')
    original_tokens = original.split(' ')
    processed_tokens = [remove_inserted_char(mt, ot)
                        for mt, ot in zip(misspelled_tokens, original_tokens)]
    sequence = ' '.join(processed_tokens)
    sequence = sequence.replace('  ', ' ')
    return sequence


def remove_additional_chars(sequence: str, correct: str):
    processed = ""
    keep_chars = correct.replace(' ', '')
    keep_i = 0
    for char in sequence:
        if keep_i < len(keep_chars) and char == keep_chars[keep_i]:
            processed += char
            keep_i += 1
        elif char == ' ' and not processed.endswith(' '):
            processed += char
    processed = processed.replace('  ', ' ').strip()
    return processed



def print_comparison(original, correct, corrupt, predicted):
    correct = remove_inserted_chars(correct, original)
    corrupt = remove_additional_chars(corrupt, correct)
    predicted = remove_additional_chars(predicted, correct)
    
    correct_i = 0
    corrupt_i = 0
    predicted_i = 0
    
    eval_sequence = ""
    
    color = None
    
    while correct_i < len(correct) or corrupt_i < len(corrupt) or predicted_i < len(predicted):
        correct_space = correct[correct_i] == ' '
        corrupt_space = corrupt[corrupt_i] == ' '
        predicted_space = predicted[predicted_i] == ' '
        if predicted_space and not corrupt_space:
            # did insert
            if correct_space:
                # true positive insertion
                color = "on_green"
            else:
                # false positive insertion
                color = "on_red"
        elif not predicted_space and corrupt_space:
            # did remove
            if correct_space:
                # false positive deletion
                color = "on_red"
            else:
                # true positive deletion
                color = "on_green"
        else:
            # did nothing
            if corrupt_space and not correct_space:
                # false negative deletion
                color = "on_yellow"
            elif not corrupt_space and correct_space:
                # false negative insertion
                color = "on_yellow"
        all_equal = correct_space == corrupt_space == predicted_space
        if all_equal or predicted_space:
            eval_sequence += colored(predicted[predicted_i], None, color)
            color = None
            predicted_i += 1
        if all_equal or corrupt_space:
            corrupt_i += 1
        if all_equal or correct_space:
            correct_i += 1
    
    print(eval_sequence)


if __name__ == "__main__":
    print_comparison("Hello world. The cat eats fish.",
                     "Hellox world. Thea ct eats fisX.",
                     "Helloxworld. Theact eat sfis X.",
                     "Hello xworld. The act eatsfi s X.")
