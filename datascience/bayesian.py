import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


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


def plot_stochastic(trace, name='stochastic', burn=0):
    df = pd.Series(trace[burn:])
    plt.figure(figsize=(15, 6))
    ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    ax.hist(df)
    median = df.quantile()
    mean = df.mean()
    ax.axvline(median, color='#A60628', linewidth=3,
               label="Median = {:0.2f}".format(median))
    ax.axvline(mean, color='#467821', linewidth=3,
               label="Mean = {:0.2f}".format(mean))
    ax.set_title(r"Histogram of {}".format(name))
    ax.set_xlabel(name)
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.axvline(df.quantile(), color='#A60628', linewidth=3, label="Median")

    ax = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    ax.plot(df)
    ax.set_title(r"Trace of {}".format(name))
    ax.set_xlabel("Steps")

    ax = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    ax.plot(autocorr_fast(df))
    ax.set_ylim(-1, 1)
    ax.set_title("Autocorrelation of {}".format(name))
    ax.set_xlabel("Lag")
