import sys

from project import src
from src.helper.files import read_sequences
from src.sequence.sequence_corruptor import SequenceCorruptor
from src.anonymous import paths as anonymous_paths

if __name__ == "__main__":
    subset = sys.argv[1]
    insert = sys.argv[2] == "1"
    delete = sys.argv[3] == "1"
    
    if subset == "test":
        sequence_file = anonymous_paths.PROJECT_TEST
    else:
        sequence_file = anonymous_paths.PROJECT_VALIDATION
    
    sequences = list(read_sequences(sequence_file))[:1000]
    
    corruptor = SequenceCorruptor(tokenization=True,
                                  insert=insert,
                                  delete=delete,
                                  n=None,
                                  p=0.05,
                                  seed=42)
    
    for sequence in sequences:
        corruptions, corrupt = corruptor.corrupt(sequence)
        print(corrupt)

