import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import math


def fit_data(data, distribution=scipy.stats.pareto, fscale=False):
    """
    Fit a pareto distribution on the data, return the parameters.
    The parameters are (b, loc, scale).
    `b` is the parameter of the Pareto distribution.
    `loc` and `scale` are for locating and scaling the data, corresponding
    to the mean and standard deviation in normal distributions.
    The default values are loc = 0.0 and scale = 1.0
    """
    if fscale:
        return distribution.fit(data, fscale=1.0)
    else:
        return distribution.fit(data)


def sample(params, size=1, distribution=scipy.stats.pareto):
    """
    Sample `size` samples from a pareto distribution given the parameters `params`.
    The parameters are `b, loc, scale` as defined earlier.
    """
    b, loc, scale = params
    return distribution.rvs(b, loc=loc, scale=scale, size=size)


def read_data(file):
    with open(file) as f:
        data = [float(line) for line in f]
    return data


def fit_resample_plot(data, distribution=scipy.stats.pareto, title=None, fscale=False):
    """
    Take a data population and fit a pareto distribution and sample
    data with the same parameters to compare the two distributions.
    """
    b, loc, scale = params = fit_data(data, distribution, fscale=fscale)
    resampled_data = sample(params, 10000, distribution)
    min_val = min(0, min(resampled_data))
    max_val = max(max(data), max(resampled_data))
    bins = [i / 100 for i in range(math.floor(min_val), math.ceil(max_val * 100) + 1)]
    sns.distplot(resampled_data, label='resampled', kde=False, bins=bins, norm_hist=True)
    sns.distplot(data, label='original', kde=False, bins=bins, norm_hist=True)
    plt.legend()
    param_name = "b" if distribution == scipy.stats.pareto else "a"
    if title is not None:
        plt.suptitle(title)
    plt.title(f"{param_name}={b:.4f}, loc={loc:.4f}, scale={scale:.4f}")
    plt.xlabel("Error rate")
    plt.ylabel("Frequency (normalized)")
    plt.show()
    print('10 values from original  data:', np.round(data[:10]))
    print('10 values from resampled data:', np.round(resampled_data[:10]))
    print(f'fitted parameters: {b},  {loc},  {scale}')


if __name__ == "__main__":
    data_path = sys.argv[1]
    data = read_data(data_path)
    exclude_zeros = "-nozero" in sys.argv
    gamma = "-gamma" in sys.argv
    power = "-power" in sys.argv
    fscale = "-fscale" in sys.argv
    print(data)
    if exclude_zeros:
        data = [x for x in data if x > 0]
    if gamma:
        distribution = scipy.stats.gamma
        distribution_name = "gamma"
    elif power:
        distribution = scipy.stats.powerlaw
        distribution_name = "powerlaw"
    else:
        distribution = scipy.stats.pareto
        distribution_name = "pareto"
    file_name = data_path.split("/")[-1]
    title = distribution_name + " " + file_name
    fit_resample_plot(np.array(data), distribution=distribution, title=title, fscale=fscale)
