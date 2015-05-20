import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.regressionplots import influence_plot


def plot_regress_analysis(model, influence=True, annotate=True):
    plt.figure(figsize=(15, 16))

    # Residuals vs Fitted
    ax = plt.subplot2grid((3, 2), (0, 0))
    ax.set_title("Residuals vs Fitted")
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    fitted = model.predict()
    residuals = model.resid
    ax.plot(fitted, residuals, marker='.', linestyle='')

    # Model non-linearity with quadratic
    polyline = np.poly1d(np.polyfit(fitted, residuals, 2))
    max_fitted = np.max(fitted)
    xs = np.append(np.arange(np.min(fitted), max_fitted), max_fitted)
    ax.plot(xs, polyline(xs), linewidth=2.5)

    # Q-Q plot
    ax = plt.subplot2grid((3, 2), (0, 1))
    ax.set_title("Q-Q")
    qqplot(model.resid_pearson, dist="norm", line='r', ax=ax)

    # Scale-Location
    ax = plt.subplot2grid((3, 2), (1, 0))
    ax.set_title("Scale-Location")
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('$|$Normalized residuals$|^{1/2}$')
    std_residuals = np.sqrt(np.abs(model.resid_pearson))
    ax.plot(fitted, std_residuals, linestyle='', marker='.')

    # Model non-linearity with quadratic
    polyline = np.poly1d(np.polyfit(fitted, std_residuals, 2))
    ax.plot(xs, polyline(xs), linewidth=2.5)

    # Residuals vs Leverage
    ax = plt.subplot2grid((3, 2), (1, 1))
    plot_leverage_resid2(model, ax, annotate=annotate)

    # Influence plot
    if influence:
        ax = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        ax = influence_plot(model, ax=ax)


def plot_leverage_resid2(results, ax, annotate=True, alpha=.05, **kwargs):
    """
    Plots leverage statistics vs. normalized residuals squared

    Parameters
    ----------
    results : results instance
        A regression results instance
    alpha : float
        Specifies the cut-off for large-standardized residuals. Residuals
        are assumed to be distributed N(0, 1) with alpha=alpha.
    ax : Axes instance
        Matplotlib Axes instance

    Returns
    -------
    fig : matplotlib Figure
        A matplotlib figure instance.
    """
    infl = results.get_influence()
    leverage = infl.hat_matrix_diag
    resid = zscore(results.resid)
    resid2 = resid ** 2
    ax.plot(resid2, leverage, '.', **kwargs)
    ax.set_xlabel("Normalized residuals**2")
    ax.set_ylabel("Leverage")
    ax.set_title("Leverage vs. Normalized residuals squared")

    _high_leverage = lambda results: 2. * (results.df_model + 1) / results.nobs

    large_leverage = leverage > _high_leverage(results)
    #norm or t here if standardized?
    cutoff = norm.ppf(1. - alpha / 2)
    large_resid = np.abs(resid) > cutoff
    labels = results.model.data.row_labels
    if labels is None:
        labels = range(results.nobs)
    index = np.where(np.logical_or(large_leverage, large_resid))[0]
    if annotate:
        ax = annotate_axes(index, labels, zip(resid2, leverage),
                           [(0, 5)] * int(results.nobs), "large",
                           ax=ax, ha="center", va="bottom")
    ax.plot(resid2[index], leverage[index], '.', **kwargs)

    ax.margins(.05, .05)
    return ax


def annotate_axes(index, labels, points, offset_points, size, ax, **kwargs):
    """
    Annotate Axes with labels, points, offset_points according to the
    given index.
    """
    for i in index:
        label = labels[i]
        point = points[i]
        offset = offset_points[i]
        ax.annotate(label, point, xytext=offset, textcoords="offset points",
                    size=size, **kwargs)
    return ax
