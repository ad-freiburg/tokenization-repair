import sys

import project
from src.helper.pickle import load_object, dump_object
from src.settings.paths import ESTIMATORS_DIR


if __name__ == "__main__":
    name = sys.argv[1]
    path = ESTIMATORS_DIR + name + "/specification.pkl"
    specification = load_object(path)
    specification.name = name
    dump_object(specification, path)
