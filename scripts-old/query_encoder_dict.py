from project import src
from src.encoding.character_encoder import get_encoder
from src.interactive.sequence_generator import interactive_sequence_generator


if __name__ == "__main__":
    encoder = get_encoder()
    for sequence in interactive_sequence_generator():
        encoded = encoder.encode_sequence(sequence)
        print(encoded)
        decoded = encoder.decode_sequence(encoded)
        print(decoded)
