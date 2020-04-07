import project
from src.ngram.tokenizer import Tokenizer
from src.interactive.sequence_generator import interactive_sequence_generator


if __name__ == "__main__":
    tokenizer = Tokenizer()
    for sequence in interactive_sequence_generator():
        tokens = tokenizer.tokenize(sequence)
        print(tokens)
