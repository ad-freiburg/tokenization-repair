import sys

from project import src
from src.corrector.threshold_holder import ThresholdHolder, FittingMethod
from src.helper.pickle import load_object
from src.settings import paths


if __name__ == "__main__":
    if "-old" in sys.argv:
        thresholds = load_object(paths.DICT_FOLDER + "new_decision_thresholds_2020_02_13.pkl")
    else:
        if "-single" in sys.argv:
            holder = ThresholdHolder(FittingMethod.SINGLE_RUN)
        else:
            holder = ThresholdHolder()
        thresholds = holder.threshold_dict
    for key in sorted(thresholds):
        print(key, thresholds[key])
