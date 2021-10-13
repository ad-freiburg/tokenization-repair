import project
from src.interactive.sequence_generator import interactive_sequence_generator
from src.postprocessing.rule_based import RuleBasedPostprocessor


if __name__ == "__main__":
    for sequence in interactive_sequence_generator():
        predicted = RuleBasedPostprocessor.correct(sequence)
        print(predicted)
