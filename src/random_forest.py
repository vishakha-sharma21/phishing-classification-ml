"""Random Forest built from the custom Gini decision tree (no scikit-learn fitting).

Training
    for each of the T trees:
        draw a bootstrap sample (n rows with replacement) — or all rows if bootstrap=False
        grow a DecisionTree that considers only ``max_features`` random features per split
Prediction
    P(phishing | x) = mean over trees of the leaf phishing frequency ("soft voting",
    as in scikit-learn); a configurable threshold turns it into a class.

Why it helps: a deep tree has low bias but high variance (small changes in the data
change its splits). Averaging T trees whose errors are only weakly correlated keeps the
bias but divides the uncorrelated part of the variance by roughly T. Bootstrapping and
random feature subsets are what make the trees different from each other; without
feature subsampling every tree would start with the same dominant split
(here ``sslfinal_state``) and the trees would be highly correlated.

Out-of-bag (OOB) estimate: each row is left out of about 36.8% of bootstrap samples;
averaging only those trees' predictions gives a validation-like estimate from
training data alone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.decision_tree import DecisionTree, _check_X


class RandomForest:
    """Bagged ensemble of randomised Gini trees for binary labels {0, 1}."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        seed: int = 42,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.seed = seed
        self.trees: list[DecisionTree] = []

    def get_params(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("n_estimators", "max_depth", "min_samples_split",
                                              "min_samples_leaf", "max_features", "bootstrap",
                                              "oob_score", "seed")}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """Train ``n_estimators`` trees; records ``fit_seconds_`` and (optionally) OOB probabilities."""
        X = _check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or not np.isin(y, (0, 1)).all():
            raise ValueError("y must be 1-D, aligned with X, and contain only 0 and 1")
        y = y.astype(int)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.n_features = X.shape[1]
        self.trees = []
        oob_sum, oob_cnt = np.zeros(n), np.zeros(n)
        t0 = time.perf_counter()
        for _ in range(self.n_estimators):
            tree_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            idx = tree_rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                seed=tree_rng).fit(X[idx], y[idx])
            self.trees.append(tree)
            if self.oob_score:
                out = np.ones(n, dtype=bool)
                out[idx] = False
                if out.any():
                    oob_sum[out] += tree.predict_proba(X[out])
                    oob_cnt[out] += 1
        self.fit_seconds_ = time.perf_counter() - t0
        if self.oob_score:
            with np.errstate(invalid="ignore", divide="ignore"):
                self.oob_proba_ = np.where(oob_cnt > 0, oob_sum / oob_cnt, np.nan)
        return self

    def _check_fitted(self) -> None:
        if not self.trees:
            raise RuntimeError("forest is not fitted")

    def predict_proba(self, X: np.ndarray, n_trees: int | None = None) -> np.ndarray:
        """Mean phishing probability over the first ``n_trees`` trees (default: all)."""
        self._check_fitted()
        X = _check_X(X, self.n_features)
        trees = self.trees[: n_trees or len(self.trees)]
        return np.mean([t.predict_proba(X) for t in trees], axis=0)

    def staged_proba(self, X: np.ndarray) -> np.ndarray:
        """Running-mean probabilities after 1, 2, ..., T trees, shape (T, n)."""
        self._check_fitted()
        X = _check_X(X, self.n_features)
        per_tree = np.array([t.predict_proba(X) for t in self.trees])
        return np.cumsum(per_tree, axis=0) / np.arange(1, len(self.trees) + 1)[:, None]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean of the trees' normalised impurity importances (as in scikit-learn)."""
        self._check_fitted()
        imp = np.mean([t.feature_importances_ for t in self.trees], axis=0)
        return imp / imp.sum() if imp.sum() > 0 else imp

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {"model": "RandomForest (custom, NumPy)", "params": self.get_params(),
                "n_features": self.n_features, "trees": [t.to_dict() for t in self.trees]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RandomForest":
        f = cls(**d["params"])
        f.n_features = d["n_features"]
        f.trees = [DecisionTree.from_dict(t) for t in d["trees"]]
        return f

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        Path(path).write_text(json.dumps({**self.to_dict(), **(extra or {})}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RandomForest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
