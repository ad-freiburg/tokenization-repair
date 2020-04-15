import project
from src.interactive.parameters import Parameter, ParameterGetter
import matplotlib.gridspec as gridspec


params = [Parameter("metric", "-m", "str",
                    help_message="Choose metric(s) from {f1, acc, t}."),
          Parameter("approaches", "-a", "str"),
          Parameter("noise_level", "-n", "float"),
          Parameter("file_name", "-o", "str")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.results_holder import Metric, ResultsHolder
from src.benchmark.benchmark import Subset, get_benchmark_name, get_error_probabilities


METRICS = {
    "f1": Metric.F1,
    "acc": Metric.SEQUENCE_ACCURACY,
    "t": Metric.MEAN_RUNTIME
}


Y_LABELS = {
    Metric.F1: "F-score",
    Metric.SEQUENCE_ACCURACY: "sequence accuracy",
    Metric.MEAN_RUNTIME: "mean runtime [sec]"
}


COLOUR_DICT = {
    "combined": "tab:blue",
    "softmax": "tab:orange",
    "sigmoid": "tab:green",
    "beam_search": "tab:red",
    "bicontext": "tab:purple",
    "google": "tab:brown",
    "enchant": "tab:pink"
}


LINE_STYLES = {
    "do_nothing": ('o', '-'),
    "greedy": ('x', '-'),
    "dynamic_bi": ('d', '-'),
    "dp_fixer": ('P', '-'),
    "google": ("$\mathrm{G}$", '-'),
    "enchant": ("$\mathrm{E}$", '-')
}


def get_plotting_style(approach: str):
    if approach in LINE_STYLES:
        marker, linestyle = LINE_STYLES[approach]
    elif approach.endswith("_robust"):
        marker = 'v'
        linestyle = "--"
    else:
        marker = 'o'
        linestyle = '-'
    colour = "gray"
    for name in COLOUR_DICT:
        if approach.startswith(name):
            colour = COLOUR_DICT[name]
    return colour, marker, linestyle


if __name__ == "__main__":
    SUBSET = Subset.TEST

    approaches = parameters["approaches"]
    metrics = parameters["metric"]
    if not isinstance(metrics, list):
        metrics = [metrics]
    metrics = [METRICS[metric] for metric in metrics]
    n_subplots = len(metrics)

    results_holder = ResultsHolder()

    error_probabilities = get_error_probabilities()

    fig = plt.figure(figsize=(2.5 + 4.8 * n_subplots, 4.8))
    fig.suptitle("typo probability = %.1f" % parameters["noise_level"])
    gs = gridspec.GridSpec(1, n_subplots)
    gs.update(wspace=0.1, hspace=0)

    for subplot, metric in enumerate(metrics):
        values = {approach: np.zeros_like(error_probabilities) for approach in approaches}
        for i, p in enumerate(error_probabilities):
            benchmark = get_benchmark_name(parameters["noise_level"], p)
            print(benchmark)
            for approach in approaches:
                if results_holder.contains(benchmark, SUBSET, approach, metric):
                    print('', approach)
                    value = results_holder.get(benchmark, SUBSET, approach, metric)
                    print(' ', metric, value)
                else:
                    value = 0
                values[approach][i] = value

        ax = plt.subplot(gs[subplot])
        x = [p if p != np.inf else 1.1 for p in error_probabilities]
        for approach in approaches:
            indices = [i for i in range(len(values[approach])) if values[approach][i] != 0]
            xx = [x[i] for i in indices]
            yy = [values[approach][i] for i in indices]
            colour, marker, linestyle = get_plotting_style(approach)
            if len(xx) > 0 and xx[-1] == 1.1:
                ax.plot(xx[:-1], yy[:-1], label=approach, color=colour, marker=marker, linestyle=linestyle)
                ax.plot(xx[-1], yy[-1], color=colour, marker=marker)
            else:
                ax.plot(xx, yy, label=approach, color=colour, marker=marker, linestyle=linestyle)

        plt.xticks(x,
                   ["%.1f" % p if p != np.inf else "no spaces" for p in error_probabilities],
                   rotation=90)

        if metric == Metric.SEQUENCE_ACCURACY:
            ax.set_ylim((0, 1.01))
            ax.set_yticks(np.arange(0, 1.1, 0.1))
        elif metric == Metric.F1:
            ax.set_ylim(0, 1.01)
            ax.set_yticks(np.arange(0, 1.1, 0.1))

        ax.set_xlabel("error probability")
        ax.set_ylabel(Y_LABELS[metric])
        box = ax.get_position()
        ax.set_position([box.x0,
                         box.y0 + 0.1,
                         box.width * 0.8,
                         box.height * 0.9])
        handles, labels = ax.get_legend_handles_labels()
    #plt.legend(loc="center left", bbox_to_anchor=(legend_x, legend_y))
    fig.legend(handles, labels, loc="center right")
    if parameters["file_name"] != "0":
        plt.savefig("plots/" + parameters["file_name"])
    plt.show()
