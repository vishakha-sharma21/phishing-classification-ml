"""Model interpretation utilities: permutation importance and stability measures.

Permutation importance shuffles ONE original feature column of the validation frame
(before encoding, so all one-hot columns of that feature move together) and records the
drop in validation ROC-AUC. It measures how much the fitted model relies on the feature
for ranking on held-out data — an association, not a causal effect. Correlated features
share credit: shuffling one of two redundant features costs little because the other
still carries the information.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_SEED


def permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    frame: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    n_repeats: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Mean and sd of the ROC-AUC drop when each feature is shuffled ``n_repeats`` times."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict(frame))
    rows = []
    for f in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[f] = rng.permutation(shuffled[f].to_numpy())
            drops.append(base - roc_auc_score(y, predict(shuffled)))
        rows.append({"feature": f, "mean_auc_drop": float(np.mean(drops)), "std": float(np.std(drops, ddof=1))})
    return pd.DataFrame(rows).sort_values("mean_auc_drop", ascending=False).reset_index(drop=True)


def rank_stability(importances: np.ndarray, names: list[str], top_k: int = 10) -> dict:
    """Agreement between importance vectors from different seeds/folds (rows = runs).

    Returns mean pairwise Spearman correlation and how often each feature is in the top-k.
    """
    importances = np.asarray(importances)
    n = len(importances)
    rhos = [spearmanr(importances[i], importances[j]).statistic for i in range(n) for j in range(i + 1, n)]
    top = [set(np.argsort(-row)[:top_k]) for row in importances]
    freq = {names[j]: sum(j in t for t in top) / n for j in range(len(names))}
    return {"mean_pairwise_spearman": float(np.mean(rhos)), "min_pairwise_spearman": float(np.min(rhos)),
            f"top{top_k}_frequency": {k: v for k, v in sorted(freq.items(), key=lambda kv: -kv[1]) if v > 0}}
