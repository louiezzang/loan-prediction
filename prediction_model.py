""" This class trains the model and predicts the loan availability.
author: Younggue Bae
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


class PredictionModel(object):

    def __init__(self,
                 solver='liblinear',
                 penalty='l2',
                 max_iter=100,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 debug=True):
        """
        Constructor.
        :param solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’)
        :param penalty: ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
        :param max_iter: Maximum number of iterations taken for the solvers to converge (default=100)
        :param tol: Tolerance for stopping criteria (default=1e-4)
        :param C: Inverse of regularization strength; must be a positive float. Like in support vector machines,
               smaller values specify stronger regularization (default=1.0)
        :param fit_intercept: Specifies if a constant (a.k.a. bias or intercept)
               should be added to the decision function (default=True)
        """
        self.debug = debug
        self.lr = LogisticRegression(solver=solver,
                                     penalty=penalty,
                                     max_iter=max_iter,
                                     tol=tol,
                                     C=C,
                                     fit_intercept=fit_intercept,
                                     random_state=0
                                     )

    def fit(self, X, y):
        """
        Fits on training data.
        :param X: pd.DataFrame
                  Input features
        :param y: np.ndarray
                  Ground truth labels as a numpy array of 0-s and 1-s.
        :return: None
        """
        return self.lr.fit(X, y)

    def predict(self, X):
        """
        Predicts class labels on new data.
        :param X: pd.DataFrame
                  Input features
        :return: np.ndarray
                 eg. np.array([1, 0, 1])
        """
        return self.lr.predict(X)

    def predict_proba(self, X):
        """
        Predicts the probability of each label.
        :param X: pd.DataFrame
                  Input features
        :return: np.ndarray
                 eg. np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        """
        return self.lr.predict_proba(X)

    def evaluate(self, X, y):
        """
        Gets the value of the following metrics: F1-score, LogLoss.
        :param X: pd.DataFrame
                  Input features
        :param y: np.ndarray
                  Ground truth labels as a numpy array of 0-s and 1-s.
        :return: dict
                 eg. {'f1_score': 0.8, 'logloss': 0.7}
        """
        # return self.lr.score(X, y)
        y_pred = self.predict(X)
        f1_score = metrics.f1_score(y, y_pred)
        log_loss = metrics.log_loss(y, y_pred)

        confmat = metrics.confusion_matrix(y, y_pred)
        print('\nConfusion matrix:')
        print(confmat)
        print('\nClassification Report:')
        print(metrics.classification_report(y, y_pred))

        return {
            'f1_score': f1_score,
            'log_loss': log_loss
        }

    def tune_parameters(self, X, y):
        """
        Runs K-fold cross validation to choose the best hyperparameters.
        Note: Output the average scores across all CV validation partitions.
        :param X: pd.DataFrame
                  Input features
        :param y: np.ndarray
                  Ground truth labels as a numpy array of 0-s and 1-s.
        :return: dict
                 eg. {'tol': 0.02, 'fit_intercept': False, 'solver': 'sag'}
        """
        print("\nTuning hyperparameters: It will take time.............")
        parameters = {
            # 'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
            'solver': ('liblinear', 'sag'),
            'C': [1, 1000],
            'tol': [1e-4, 0.001, 0.02],
            'fit_intercept': [True, False],
            # 'penalty': ['l1', 'l2']
        }

        lr = LogisticRegression(max_iter=1000, random_state=0)
        gs = GridSearchCV(estimator=lr,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv=5,
                          n_jobs=-1
                          )
        gs.fit(X, y)

        # scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
        # print('CV accuracy =', np.mean(scores))

        if self.debug:
            print('best score =', gs.best_score_)
            print('best params =', gs.best_params_)
        return gs.best_params_

