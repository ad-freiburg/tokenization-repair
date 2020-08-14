import project
from src.evaluation.tolerant import tolerant_preprocess_sequences


if __name__ == "__main__":
    original = """Popeștii de Jos is a village in Drochia District, Moldova."""
    correct = """Popeștii do Jose is v village in Drochiam District, Moldova."""
    corrupt = """Popeș tii d o Jo se is vvillage in Drochiam District ,Moldova."""
    predicted = """Popeștii do Jose is v village in Drochiam District, Moldova."""
    correct, corrupt, predicted = tolerant_preprocess_sequences(original, correct, corrupt, predicted)
    print(correct)
    print(corrupt)
    print(predicted)

    original = "the heavenly Jerusalem"
    correct = "the hheavenly Jerusalem"
    corrupt = "the hheavenly Jerusalem"
    predicted = "theh heavenly Jerusalem"
    correct, corrupt, predicted = tolerant_preprocess_sequences(original, correct, corrupt, predicted)
    print(correct)
    print(corrupt)
    print(predicted)
