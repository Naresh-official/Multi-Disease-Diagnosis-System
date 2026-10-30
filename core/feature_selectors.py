"""Custom feature selectors for disease models."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


class RankedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects a slice of features ranked by ANOVA F-value on the training set.
    Allows calibrating feature selection to informative biomarker panels
    without overfitting hyper-separable outliers in high-dimensional microarrays.
    """

    def __init__(self, start_rank: int = 0, n_features: int = 100):
        self.start_rank = start_rank
        self.n_features = n_features
        self.selected_indices_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        f_vals, _ = f_classif(X, y)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        sorted_indices = np.argsort(f_vals)[::-1]
        self.selected_indices_ = sorted_indices[
            self.start_rank : self.start_rank + self.n_features
        ]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def get_support(self):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask
