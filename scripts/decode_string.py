import project
from src.interactive.sequence_generator import interactive_sequence_generator


if __name__ == "__main__":
    for sequence in interactive_sequence_generator():
        for char in sequence:
            print(char.encode())