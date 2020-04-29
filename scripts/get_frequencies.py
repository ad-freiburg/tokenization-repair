import project
from src.ngram.tokenizer import Tokenizer
from src.ngram.unigram_holder import UnigramHolder
from src.ngram.bigram_holder import BigramHolder
from src.interactive.sequence_generator import interactive_sequence_generator


if __name__ == "__main__":
    tokenizer = Tokenizer()
    unigrams = UnigramHolder()
    print("%i unigrams" % len(unigrams))
    bigrams = BigramHolder.load()
    print("%i bigrams" % len(bigrams))

    for sequence in interactive_sequence_generator():
        tokens = tokenizer.tokenize(sequence)
        texts = [token.text for token in tokens]
        for text in texts:
            print(text, unigrams.get(text))
        for bigram in zip(texts, texts[1:]):
            print(bigram, bigrams.get(bigram))
