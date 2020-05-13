import sys

import project
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.evaluation.evaluator import Evaluator
from src.datasets.wikipedia import Wikipedia
from src.helper.data_structures import izip
from src.evaluation.print_methods import print_evaluator


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
    return ' '.join(processed_tokens)


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
    return processed.strip()


if __name__ == "__main__":
    benchmark_name = sys.argv[1]
    predictions_file_name = sys.argv[2]
    subset = Subset.TEST if benchmark_name == "doval" else Subset.DEVELOPMENT

    benchmark = Benchmark(benchmark_name, subset)
    original_sequences = Wikipedia.development_sequences()
    sequence_pairs = benchmark.get_sequence_pairs(BenchmarkFiles.CORRUPT)
    if predictions_file_name == "corrupt.txt":
        predicted_sequences = [corrupt for _, corrupt in sequence_pairs]
    else:
        predicted_sequences = benchmark.get_predicted_sequences(predictions_file_name)

    evaluator = Evaluator()

    for s_i, original, (correct, corrupt), predicted in izip(original_sequences, sequence_pairs, predicted_sequences):
        print(original)
        print(correct)
        print(corrupt)
        print(predicted)

        correct_processed = remove_inserted_chars(correct, original).replace('  ', ' ')
        corrupt_processed = remove_additional_chars(corrupt, correct_processed).replace('  ', ' ')
        predicted_processed = remove_additional_chars(predicted, correct_processed).replace('  ', ' ')

        print(correct_processed)
        print(corrupt_processed)
        print(predicted_processed)

        evaluator.evaluate(predictions_file_name,
                           s_i,
                           original_sequence=correct_processed,
                           corrupt_sequence=corrupt_processed,
                           predicted_sequence=predicted_processed,
                           evaluate_ed=False)
        evaluator.print_sequence()

        print()

    print_evaluator(evaluator)
