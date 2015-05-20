import numpy as np
import statsmodels.formula.api as sm
from sklearn.cross_validation import KFold


def kfold(df, formula, predictors, score_fct, n_folds=10):
    k_fold = KFold(len(df), n_folds)
    scores = []
    for train, test in k_fold:
        training, test = df.ix[train], df.ix[test]
        model = sm.ols(formula, training).fit()
        scores.append(
            score_fct(test.order_count, model.predict(test[predictors])))
    return np.mean(scores)
