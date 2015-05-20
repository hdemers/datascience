import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.learning_curve import learning_curve as lc
from sklearn.cross_validation import cross_val_score


def learning_curve_mse(clf, data, predictors, target, train_ratios=None, cv=5):
    palette = sns.color_palette()
    if train_ratios is None:
        train_ratios = np.arange(0.1, 1.1, 0.1)

    train_sizes, train_scores, test_scores = lc(
        clf,
        data[predictors], data[target].values.ravel(),
        train_sizes=train_ratios,
        scoring='mean_squared_error', cv=cv)

    train_scores = -train_scores
    test_scores = -test_scores
    df = pd.DataFrame({'training error': train_scores.mean(axis=1),
                       'test error': test_scores.mean(axis=1),
                       'training_std': train_scores.std(axis=1),
                       'test_std': test_scores.std(axis=1)},
                      index=(train_ratios * 100.))
    df.index.name = 'percent'

    ax = df[['training error', 'test error']].plot(figsize=(9, 5),
                                                   xlim=(0, 105), marker='o')
    ax.fill_between(df.index, df['training error'] - df.training_std,
                    df['training error'] + df.training_std, alpha=0.2)
    ax.fill_between(df.index, df['test error'] - df.test_std,
                    df['test error'] + df.test_std, alpha=0.2,
                    color=palette[1])
    ax.set_xlabel("Training set size in percent")
    ax.set_ylabel("MSE")
    ax.set_title("Learning curves")
    return df


def cv_mse(clf, data, predictors, target, cv=5):
    mse = cross_val_score(clf,
                          data[predictors].values,
                          data[target].values.ravel(),
                          cv=cv,
                          scoring='mean_squared_error').mean()
    return -mse
