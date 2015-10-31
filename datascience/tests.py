from numpy.random import permutation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def permutation_test(observed, expected, stat_func, n_samples=10000):
    all_samples = observed.append(expected)
    n_obs = len(observed)
    values = []
    for i in range(n_samples):
        samples = permutation(all_samples)
        values.append(stat_func(samples[:n_obs]) - stat_func(samples[n_obs:]))
    return values


def significance(dist1, dist2, stat_func, n_samples=10000, graph=True):
    palette = sns.color_palette()
    deltas = permutation_test(dist1, dist2, stat_func, n_samples)
    obs_delta = abs(stat_func(dist1) - stat_func(dist2))

    s = pd.Series(deltas)
    p_value = ((s < -obs_delta).sum() + (s > obs_delta).sum()) / float(len(s))

    if graph:
        fig, ax = plt.subplots(1, figsize=(8, 5))
        sns.distplot(deltas, ax=ax)
        ax.axvline(obs_delta, color=palette[0])

        ax.text(0.05, 0.9, "delta: {}".format(obs_delta),
                transform=ax.transAxes, fontsize=13, color="0.4")

        ax.text(0.05, 0.82, "p_value: {:0.4f}".format(p_value),
                transform=ax.transAxes, fontsize=13, color="0.4")
        ax.set_title("Permutation test results", fontsize=14)
    return obs_delta, p_value
