"""Metrics and resampling-based uncertainty. Positive class = phishing (1).

Ranking metrics (ROC-AUC, PR-AUC) and log-loss always use predicted phishing
probabilities, never hard labels. Threshold metrics use ``proba >= threshold``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import RANDOM_SEED

PROBA_CLIP = 1e-6  # log-loss clip so one over-confident leaf cannot dominate


def clipped_log_loss(y: np.ndarray, proba: np.ndarray) -> float:
    """Binary log-loss with probabilities clipped to [1e-6, 1 - 1e-6]."""
    p = np.clip(proba, PROBA_CLIP, 1 - PROBA_CLIP)
    return float(log_loss(y, p, labels=[0, 1]))


def compute_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """All threshold and ranking metrics for one set of predictions."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    if y.shape != proba.shape:
        raise ValueError("y and proba must have the same shape")
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if both else float("nan"),
        "log_loss": clipped_log_loss(y, proba),
        "brier": float(brier_score_loss(y, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


KEY_METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc")


def bootstrap_ci(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
    metrics: tuple[str, ...] = KEY_METRICS,
) -> dict[str, dict[str, float]]:
    """Percentile 95% intervals from resampling rows of a fixed evaluation set."""
    rng = np.random.default_rng(seed)
    y, proba = np.asarray(y), np.asarray(proba)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = compute_metrics(y[idx], proba[idx], threshold)
        for k in metrics:
            samples[k].append(m[k])
    point = compute_metrics(y, proba, threshold)
    return {k: {"point": point[k], "low": float(np.percentile(v, 2.5)),
                "high": float(np.percentile(v, 97.5))} for k, v in samples.items()}


def paired_bootstrap_diff(
    y: np.ndarray,
    proba_a: np.ndarray,
    thr_a: float,
    proba_b: np.ndarray,
    thr_b: float,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
    metrics: tuple[str, ...] = ("f1", "roc_auc", "pr_auc", "accuracy"),
) -> dict[str, dict[str, float]]:
    """Distribution of metric(A) - metric(B) when both are scored on the same resampled rows.

    Pairing removes the variance shared by both models, so this is more informative
    than comparing two separate intervals. ``p_le_0`` is the share of resamples where
    A is not better; it is a descriptive quantity, not a formal p-value.
    """
    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        ma = compute_metrics(y[idx], proba_a[idx], thr_a)
        mb = compute_metrics(y[idx], proba_b[idx], thr_b)
        for k in metrics:
            diffs[k].append(ma[k] - mb[k])
    out = {}
    for k, v in diffs.items():
        v = np.asarray(v)
        out[k] = {"mean": float(v.mean()), "low": float(np.percentile(v, 2.5)),
                  "high": float(np.percentile(v, 97.5)), "p_le_0": float((v <= 0).mean())}
    return out


def verdict(diff: dict[str, float], small: float = 0.01) -> str:
    """Plain-language verdict on a paired difference A - B (for higher-is-better metrics)."""
    lo, hi, mean = diff["low"], diff["high"], diff["mean"]
    if lo > 0:
        return "clear improvement" if mean >= small else "small but consistent improvement"
    if hi < 0:
        return "worse performance"
    if abs(mean) < small / 2:
        return "no meaningful difference"
    return "small or uncertain difference"
