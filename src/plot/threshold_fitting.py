import numpy as np
import matplotlib.pyplot as plt

from src.settings import paths


def plot_precision_recall_f1(thresholds, precision, recall, f1, sweet_spot=True, title=None, save=False):
    plt.plot(thresholds, precision, color="black", linestyle="--", label="precision")
    plt.plot(thresholds, recall, color="black", linestyle=":", label="recall")
    plt.plot(thresholds, f1, color="black", label="f1")
    if sweet_spot:
        sweet = np.argmax(f1)
        plt.plot(thresholds[sweet], f1[sweet], fillstyle="none", marker="o", color="black", markersize=8)
    plt.legend(loc="best")
    plt.xlabel("threshold")
    if title is not None:
        plt.title(title)
    if save:
        plt.savefig(paths.PLOT_DIR + title + ".png")
    else:
        plt.show()
