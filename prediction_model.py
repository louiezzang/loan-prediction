""" This class trains the model and predicts the loan availability.
author: Younggue Bae
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class PredictionModel(object):

    def __init__(self, solver='liblinear', penalty='l2', max_iter=100, debug=True):
        """
        Constructor.
        :param solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’)
        :param penalty: ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
        :param max_iter: Maximum number of iterations taken for the solvers to converge (default=100)
        :param debug:
        """
        self.debug = debug
        self.lr = LogisticRegression(solver=solver, penalty=penalty, C=1000.0, max_iter=max_iter, random_state=0)

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
        return self.predict(X)

    def predict_proba(self, X):
        """
        Predicts the probability of each label.
        :param X: pd.DataFrame
                  Input features
        :return: np.ndarray
                 eg. np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        """
        return self.predict_proba(X)

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

    def tune_parameters(self, X, y):
        """
        Runs K-fold cross validation to choose the best parameters.
        Note: Output the average scores across all CV validation partitions.
        :param X: pd.DataFrame
                  Input features
        :param y: np.ndarray
                  Ground truth labels as a numpy array of 0-s and 1-s.
        :return: dict
                 eg. {'tol': 0.02, 'fit_intercept': False, 'solver': 'sag'}
        """