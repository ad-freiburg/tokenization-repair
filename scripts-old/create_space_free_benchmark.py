import project
from src.helper.files import read_lines, write_lines
from src.settings import paths

if __name__ == "__main__":
    path = paths.BENCHMARKS_DIR + "doval/test/"
    corrupt_sequences = [line.replace(' ', '') for line in read_lines(path + "correct.txt")]
    write_lines(path + "corrupt.txt", corrupt_sequences)
