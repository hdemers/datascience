from numpy.random import permutation


def permutation_test(observed, expected, stat_func, n_samples=10000):
    all_samples = observed.append(expected)
    n_obs = len(observed)
    values = []
    for i in range(n_samples):
        samples = permutation(all_samples)
        values.append(stat_func(samples[:n_obs], samples[n_obs:]))
    return values
