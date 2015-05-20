from statsmodels.stats.outliers_influence import variance_inflation_factor


def pprint(array, labels, format_string='>6.2f'):
    f = max(len(l) for l in labels)
    for l, e in zip(labels, array):
        print "{:{f}} {:{fs}}".format(l, e, f=f, fs=format_string)


def variance_inflation_factors(df, labels):
    df = df[labels]
    fill = max([len(name) for name in df.columns])
    for index, name in enumerate(df.columns):
        vif = variance_inflation_factor(df.values, index)
        print "{:{fill}} {:>7.1f}".format(name, vif, fill=fill)
