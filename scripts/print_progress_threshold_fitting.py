import project
from src.settings import paths
from src.helper.files import get_files
from src.helper.pickle import load_object


if __name__ == "__main__":
    for file in sorted(get_files(paths.THRESHOLD_FITTER_DIR)):
        fitter = load_object(paths.THRESHOLD_FITTER_DIR + file)
        print(file, fitter.n_sequences)
