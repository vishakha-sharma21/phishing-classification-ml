"""Baselines and scikit-learn REFERENCE models.

Nothing here is used inside the custom implementations; these exist only to give
honest points of comparison. Every result produced from this module is labelled
"reference" or "baseline" in reports.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.tree import DecisionTreeClassifier


class MajorityBaseline:
    """Predicts the training majority class; probability = training phishing rate."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaseline":
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.prior_)


class SingleFeatureBaseline:
    """Scores a row by the Laplace-smoothed training phishing rate of one feature's value."""

    def __init__(self, feature_index: int, alpha: float = 1.0) -> None:
        self.feature_index = feature_index
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleFeatureBaseline":
        x = X[:, self.feature_index]
        self.prior_ = float(np.mean(y))
        self.rates_ = {
            float(v): (y[x == v].sum() + self.alpha * self.prior_) / ((x == v).sum() + self.alpha)
            for v in np.unique(x)
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.rates_.get(float(v), self.prior_) for v in X[:, self.feature_index]])


def sklearn_logistic(l2: float, n_train: int, seed: int = 42) -> SkLogisticRegression:
    """sklearn LR matched to the custom objective mean(BCE) + (l2/2)||w||^2.

    sklearn minimises C * sum(BCE) + 0.5 ||w||^2, so C = 1 / (l2 * n). ``l2 = 0``
    maps to no penalty.
    """
    if l2 == 0:
        return SkLogisticRegression(penalty=None, max_iter=5000, random_state=seed)
    return SkLogisticRegression(C=1.0 / (l2 * n_train), max_iter=5000, random_state=seed)


def sklearn_forest(params: dict, seed: int = 42) -> RandomForestClassifier:
    """sklearn RF with the custom forest's hyperparameter names mapped across."""
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        min_samples_split=params.get("min_samples_split", 2),
        max_features=params.get("max_features", "sqrt"),
        bootstrap=params.get("bootstrap", True),
        random_state=seed,
        n_jobs=1,
    )


def sklearn_tree(params: dict, seed: int = 42) -> DecisionTreeClassifier:
    """sklearn CART tree with matching hyperparameters (Gini criterion)."""
    return DecisionTreeClassifier(
        max_depth=params.get("max_depth"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        min_samples_split=params.get("min_samples_split", 2),
        max_features=params.get("max_features"),
        random_state=seed,
    )


def sk_proba(model, X: np.ndarray) -> np.ndarray:
    """Phishing-class probability from a fitted sklearn classifier."""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]
