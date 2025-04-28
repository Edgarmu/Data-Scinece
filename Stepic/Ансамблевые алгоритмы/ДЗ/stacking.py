import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from joblib import Parallel, delayed

# from mlxtend.plotting import plot_decision_regions


class Stacking:
    def __init__(self, estimators, final_estimator, blending=False, cv=5, n_jobs=-1):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.blending = blending
        self.cv = cv
        self.n_jobs = n_jobs

    def _X_pred(self, estimator, data):
        if self.blending:
            X_train_v, y_train_v, X_val = data
            return estimator.fit(X_train_v, y_train_v).predict(X_val)
        else:
            X_train, y_train = data
            return cross_val_predict(estimator, X_train, y_train, cv=self.cv)

    def _X_test_pred(self, estimator, data):
        X_train, y_train, X_test = data

        return estimator.fit(X_train, y_train).predict(X_test)

    def _meta_data(self, X_train, y_train, X_test):
        if self.blending:
            # used hold-out cross-validation
            X_train_v, X_val, y_train_v, y_val = train_test_split(
                X_train, y_train, random_state=0
            )
            train_data = [X_train_v, y_train_v, X_val]
            test_data = [X_train_v, y_train_v, X_test]
            meta_y_train = y_val
        else:
            train_data = [X_train, y_train]
            test_data = [X_train, y_train, X_test]
            meta_y_train = y_train

        cv_X_train_preds = (
            delayed(self._X_pred)(est, train_data) for est in self.estimators
        )
        X_test_preds = (
            delayed(self._X_test_pred)(est, test_data) for est in self.estimators
        )

        meta_X_train = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(cv_X_train_preds))
        meta_X_test = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(X_test_preds))

        return meta_X_train.T, meta_y_train, meta_X_test.T

    def fit_predict(self, X_train, y_train, X_test):
        # meta learner or blender
        meta_X_train, meta_y_train, meta_X_test = self._meta_data(
            X_train, y_train, X_test
        )

        return self.final_estimator.fit(meta_X_train, meta_y_train).predict(meta_X_test)
