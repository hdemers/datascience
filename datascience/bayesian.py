import math

import numpy as np
import pandas as pd
from scipy import signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.color_palette()
title_size = 15
label_size = 12
textcolor = "0.3"
round_to_n = lambda x, n: round(x,
                                -int(math.floor(math.log10(abs(x)))) + (n - 1))


def autocorr(x):
    unbiased = x - np.mean(x)
    norm = np.sum(unbiased ** 2)
    acorr = np.correlate(unbiased, unbiased, "full") / norm
    # use only second half
    return acorr[len(acorr) / 2:]


def autocorr_fast(x):
    unbiased = x - np.mean(x)
    norm = np.sum(unbiased ** 2)
    acorr = signal.fftconvolve(unbiased, unbiased[::-1], mode='full') / norm
    # use only second half
    return acorr[len(acorr) / 2:]


def plot_stochastic(node, chain=None, burn=0):
    name = node.__name__
    trace = node.trace(chain=chain)[burn:]
    s = pd.Series(trace)

    plt.figure(figsize=(15, 6))
    ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    histogram(node, chain, burn, ax)

    ax = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    ax.plot(s)
    ax.set_title(r"Trace of {}".format(name), size=title_size)
    ax.set_xlabel("Steps", size=label_size)

    ax = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    ax.plot(autocorr_fast(s))
    ax.set_ylim(-1, 1)
    ax.set_title("Autocorrelation of {}".format(name), size=title_size)
    ax.set_xlabel("Lag", size=label_size)


def histogram(node, chain=None, burn=0, ax=None):
    name = node.__name__
    trace = node.trace(chain=chain)[burn:]
    s = pd.Series(trace)

    ax = sns.distplot(s, ax=ax)
    ax.set_title(r"Histogram of {}".format(name), size=title_size)
    ax.set_xlabel(name, size=label_size)
    ax.set_yticklabels([''])

    # Print the mode and median
    median = s.quantile()
    ax.text(0.5, 0.9, "mode = {}".format(round_to_n(mode(trace), 3)),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, size=17, color=textcolor)
    ax.text(0.5, 0.83, "median = {}".format(round_to_n(median, 3)),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, size=17, color=textcolor)

    mcmc_stats = node.stats(chain=chain)

    # Show HDI
    hdi_min, hdi_max = mcmc_stats['95% HPD interval']
    y_min, y_max = ax.get_ylim()
    y = 0.05 * (y_max - y_min) + y_min
    ax.plot((hdi_min, hdi_max), (y, y), color=textcolor, linewidth=4)
    # Write the HDI limits
    y = 0.08 * (y_max - y_min) + y_min
    ax.text(hdi_min, y, round_to_n(hdi_min, 3), size=12, color=textcolor,
            verticalalignment='center', horizontalalignment='center')
    ax.text(hdi_max, y, round_to_n(hdi_max, 3), size=12, color=textcolor,
            verticalalignment='center', horizontalalignment='center')
    # Write 95% HDI
    x = 0.5 * (hdi_max - hdi_min) + hdi_min
    y = 0.1 * (y_max - y_min) + y_min
    ax.text(x, y, "95% HDI", horizontalalignment='center',
            color=textcolor, size=16)


def hdi(trace, cred_mass=0.95):
    s = np.sort(trace)
    ci_index = int(np.ceil(cred_mass * len(s)))
    n_cis = len(s) - ci_index

    ci_widths = s[-n_cis:] - s[:n_cis]
    argmin_width = np.argmin(ci_widths)
    return s[argmin_width], s[argmin_width + ci_index]


def mode(trace):
    s = pd.Series(trace)
    kernel = stats.gaussian_kde(s)
    x = np.linspace(s.min(), s.max(), 1000)
    density = pd.Series(kernel.evaluate(x), index=x)
    return density.argmax()


def representativeness(model_variables, stochastic, steps=5000):
    pass
