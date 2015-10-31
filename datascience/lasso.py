import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import lars_path


def plot_lassolarscv_paths(clf, df, predictors, target):
    # Compute the coefficients path and plot them as a function of alpha.
    alphas, _, coefs = lars_path(df[predictors].values, df[target].values)
    coef_path = (
        pd.DataFrame(coefs.T, columns=predictors)
        .assign(alpha=alphas)
        .sort('alpha')
    )
    with sns.color_palette('deep', 12):
        ax = coef_path.plot('alpha', figsize=(11, 5), grid=False)
    ax.set_ylabel("Coefficient values")

    # Plot the MSE as a function of alpha.
    twin_ax = ax.twinx()
    mse_path = pd.DataFrame({'MSE': np.mean(clf.cv_mse_path_, axis=1),
                             'alpha': clf.cv_alphas_}).sort('alpha')

    mse_path.plot('alpha', figsize=(11, 5), linestyle='--', grid=False,
                  color='grey', ax=twin_ax)

    # Find which MSE correspond to the chosen alpha, i.e. the min MSE.
    alpha = clf.alpha_
    index = np.where(clf.cv_alphas_ == alpha)[0][0]
    mse = np.mean(clf.cv_mse_path_[index])

    twin_ax.plot([alpha], [mse], marker='o', color='grey')
    twin_ax.axvline(alpha, color='grey', linestyle='--', linewidth=1)
    twin_ax.set_ylabel("MSE")

    ax.set_title("MSE and coefficient paths as a function of alpha")

    annot = r"$\alpha$ = {:0.3f}, CV MSE: {:0.2f}".format(
        alpha, mse)
    ax.text(0.8, 0.05, annot, horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transAxes)
