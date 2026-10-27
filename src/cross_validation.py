"""Stratified k-fold cross-validation on the TRAINING partition only.

Folds come from scikit-learn's ``StratifiedKFold`` (index generation only); every model
is fitted by its own ``fit`` method. Validation and test partitions are never touched.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_SEED
from src.evaluation import compute_metrics

CV_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss")


def stratified_folds(y: np.ndarray, k: int = 5, seed: int = RANDOM_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k (fit_idx, held_out_idx) pairs with class proportions preserved."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def cross_validate(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.5,
) -> dict[str, list[float]]:
    """Per-fold metrics. ``fit_predict(X_fit, y_fit, X_held)`` must return phishing probabilities."""
    out: dict[str, list[float]] = {m: [] for m in CV_METRICS}
    for fit_idx, held_idx in folds:
        p = fit_predict(X[fit_idx], y[fit_idx], X[held_idx])
        m = compute_metrics(y[held_idx], p, threshold)
        for k in CV_METRICS:
            out[k].append(m[k])
    return out


def summarise(per_fold: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation across folds."""
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for k, v in per_fold.items()}


def paired_fold_difference(a: dict[str, list[float]], b: dict[str, list[float]], metric: str) -> dict[str, float]:
    """Mean and range of per-fold differences A - B (same folds, so the comparison is paired)."""
    d = np.asarray(a[metric]) - np.asarray(b[metric])
    return {"mean": float(d.mean()), "std": float(d.std(ddof=1)), "min": float(d.min()), "max": float(d.max()),
            "folds_A_better": int((d > 0).sum() if metric != "log_loss" else (d < 0).sum())}
