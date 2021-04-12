from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.helper.files import write_file


def save_histogram_data(data, path):
    data_str = "\n".join(str(val) for val in sorted(data))
    write_file(path, data_str)


def _plot_histogram(data,
                    bins,
                    title: str,
                    subtitle: str,
                    xlabel: str,
                    ylabel: str,
                    save_path: Optional[str],
                    exclude_zero: bool):
    if exclude_zero:
        bins = bins[1:]
        if save_path is not None:
            save_path = save_path[:-4] + "_no_zeros" + save_path[-4:]
    plt.hist(data, bins=bins)
    plt.suptitle(title)
    plt.title(subtitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


RATE_BIN_WIDTH = 0.02


def plot_rate_histogram(data: List[float],
                        title: str,
                        subtitle: str,
                        xlabel: str,
                        ylabel: str = "Frequency",
                        save_path: Optional[str] = None,
                        exclude_zero: bool = False):
    bins = [-RATE_BIN_WIDTH, 1e-16] + list(np.arange(RATE_BIN_WIDTH, 1 + RATE_BIN_WIDTH, RATE_BIN_WIDTH))
    _plot_histogram(data, bins, title, subtitle, xlabel, ylabel, save_path, exclude_zero)


def plot_histogram(data: List[int],
                   title: str,
                   subtitle: str,
                   xlabel: str,
                   ylabel: str = "Frequency",
                   save_path: Optional[str] = None,
                   exclude_zero: bool = False):
    bins = list(range(max(data) + 1))
    _plot_histogram(data, bins, title, subtitle, xlabel, ylabel, save_path, exclude_zero)
