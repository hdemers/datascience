import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score, roc_curve, auc)


def confusion(y_true=None, y_pred=None, model=None, labels=None,
              threshold=0.5):
    """ model is a statsmodels mo0del result.
    """
    if model:
        matrix = model.pred_table(threshold)
    elif y_true is not None and y_pred is not None:
        matrix = confusion_matrix(y_true, y_pred)
    else:
        raise Exception("Must provide y_true and y_pred, or model")

    df = pd.DataFrame(matrix, columns=labels, index=labels)

    df.loc['total', :] = df.sum()
    df.loc[:, 'total'] = df.sum(axis=1)

    df.columns = pd.MultiIndex.from_product([['predicted'], df.columns.values])
    df.index = pd.MultiIndex.from_product([['observed'], df.index.values])
    return df


def metrics(y_true, y_pred, labels=None):
    array = np.array(precision_recall_fscore_support(y_true, y_pred))
    df = pd.DataFrame(array.T, index=labels)

    avg_total = df.iloc[:, :-1].mean()
    avg_total.name = 'avg / total'
    avg_total.ix[3] = df.iloc[:, -1].sum()

    columns = ['precision', 'recall', 'f1-score', 'support']
    df = df.append(avg_total)
    df.columns = columns
    return df, accuracy_score(y_true, y_pred)


def lda_stats(clf, labels=None, predictors=None):
    coefs = pd.DataFrame(clf.coef_.T, index=labels, columns=['coef'])
    means = pd.DataFrame(clf.means_, columns=predictors, index=labels)
    priors = pd.DataFrame(clf.priors_, index=labels, columns=['prior'])
    return priors, means, coefs


def qda_stats(clf, labels=None, predictors=None):
    means = pd.DataFrame(clf.means_, columns=predictors, index=labels)
    priors = pd.DataFrame(clf.priors_, index=labels, columns=['prior'])
    return priors, means


def roc(y_true, y_pred_proba, thresholds=0.5):
    fpr, tpr, thres = roc_curve(y_true, y_pred_proba)

    sns.set_style('white')
    palette = sns.color_palette()

    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thres})
    ax = df.plot('fpr', 'tpr', kind='line', legend=False, grid=False)
    sns.despine(ax=ax, offset=10, trim=True)

    ax.set_xlim(-0.005, 1)
    ax.set_ylim(0., 1.005)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title('ROC')
    ax.text(0.95, 0.01, 'Area = {:0.3}'.format(auc(fpr, tpr)),
            horizontalalignment='right')

    # Add a point for a specific threshold
    def show_threshold(threshold):
        matrix = confusion_matrix(y_true, y_pred_proba > threshold)
        totals = matrix.sum(axis=1)
        fpr_tr = matrix[0, 1] / float(totals[0])
        tpr_tr = matrix[1, 1] / float(totals[1])
        ax.scatter([fpr_tr], [tpr_tr], color=palette[2], s=50, zorder=10)
        ax.text(fpr_tr + 0.015, tpr_tr, "{:.0f}%".format(threshold * 100),
                horizontalalignment='left', verticalalignment='top')

    if not isinstance(thresholds, list):
        thresholds = [thresholds]
    for threshold in thresholds:
        show_threshold(threshold)

    return df
