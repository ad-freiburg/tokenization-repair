from typing import Dict
import sys
import numpy as np

from project import src
from src.load.load_char_lm import load_char_lm
from src.encoding.character_encoder import CharacterEncoder
from src.interactive.sequence_generator import interactive_sequence_generator
from src.helper.data_structures import top_k_indices_sorted


K = 5


def print_probabilities(result: Dict, encoder: CharacterEncoder):
    print("PROBABILITIES:")
    labels = result["labels"]
    probabilities = result["probabilities"]
    logprob = 0
    for i, label in enumerate(labels):
        p = probabilities[i, :]
        logprob += np.log(p[label])
        print_str = "%s %f |" % (encoder.decode_label(label), p[label])
        top = top_k_indices_sorted(p, K)
        for t in top:
            print_str += " %s %f" % (encoder.decode_label(t), p[t])
        print(print_str)
    print("logprob = %f" % logprob)


def print_insertion_probabilities(result: Dict, encoder: CharacterEncoder):
    print("\nINSERTION PROBABILITIES:")
    labels = result["labels"]
    probabilities = result["insertion_probabilities"]
    for i in range(len(labels) + 1):
        if i > 0:
            print(encoder.decode_label(labels[i - 1]))
        p = probabilities[i]
        top = top_k_indices_sorted(p, K)
        print_str = "> " + " ".join("%s %f" % (encoder.decode_label(t), p[t]) for t in top)
        print(print_str)


def print_deletion_probabilities(result: Dict, encoder: CharacterEncoder):
    print("\nDELETION PROBABILITIES:")
    labels = result["labels"]
    probabilities = result["deletion_probabilities"]
    for i, label in enumerate(labels):
        p = probabilities[i]
        print_str = "%s %f" % (encoder.decode_label(label), p)
        print(print_str)


if __name__ == "__main__":
    model = load_char_lm(model_type=sys.argv[1],
                         model_name=sys.argv[2],
                         bwd_model_name=sys.argv[3] if len(sys.argv) > 3 else None)

    encoder = model.get_encoder()

    for sequence in interactive_sequence_generator():
        result = model.predict(sequence)
        print_probabilities(result, encoder)
        if "insertion_probabilities" in result:
            print_insertion_probabilities(result, encoder)
        if "deletion_probabilities" in result:
            print_deletion_probabilities(result, encoder)
