import project
from src.postprocessing.bigram import BigramPostprocessor
from src.interactive.sequence_generator import interactive_sequence_generator


if __name__ == "__main__":
    postprocessor = BigramPostprocessor()
    print("%i unigrams" % len(postprocessor.unigrams))
    print("%i bigrams" % len(postprocessor.bigrams))
    for sequence in interactive_sequence_generator():
        predicted = postprocessor.correct(sequence)
        print(predicted)
