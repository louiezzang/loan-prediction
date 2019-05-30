""" This provides the feature importance by using Random forests.
author: Younggue Bae
"""

from sklearn.ensemble import RandomForestRegressor


def feature_importance(X, y):
    clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
    model = clf.fit(X, y)

    values = sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1] * -1)
    return values

