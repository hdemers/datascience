import pandas as pd

from datascience.modelperf import cv_mse


def cv_max_features(clf, data, max_features, predictors, target, cv=5):
    mses = []
    n_features = range(1, max_features)
    for n_feature in n_features:
        clf.max_features = n_feature
        mses.append(cv_mse(clf, data, predictors, target, cv))
    return pd.DataFrame({'mse': mses}, index=n_features)


def cv_n_estimators(clf, data, max_estimators, predictors, target,
                    cv=5, step=1):
    mses = []
    n_estimators = range(1, max_estimators, step)
    for n_estimator in n_estimators:
        clf.n_estimators = n_estimator
        mses.append(cv_mse(clf, data, predictors, target, cv))
    return pd.DataFrame({'mse': mses}, index=n_estimators)
