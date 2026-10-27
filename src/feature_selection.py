"""Transparent feature-selection methods, fitted on the TRAINING partition only.

1. ``near_constant_filter`` — label-free. Raw variance is not comparable across these
   features (``redirect`` is coded {0,1}, so its variance is capped at 0.25 while ±1
   features reach 1), so the filter uses the share of the most frequent level instead:
   a feature whose dominant level covers ≥ τ of training rows carries little variation.
   Raw variances are still reported for reference.
2. ``mutual_information_ranking`` — supervised, training labels only. Ranks features by
   mutual information (bits) with the label; the top-k are kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def dominant_level_share(train: pd.DataFrame, features: list[str]) -> pd.Series:
    """Share of training rows taken by each feature's most frequent level."""
    return train[features].apply(lambda c: c.value_counts(normalize=True).max())


def near_constant_filter(train: pd.DataFrame, features: list[str], tau: float) -> list[str]:
    """Features kept: those whose dominant level covers < ``tau`` of training rows."""
    share = dominant_level_share(train, features)
    return [f for f in features if share[f] < tau]


def mutual_information_ranking(train: pd.DataFrame, features: list[str], target: str) -> pd.Series:
    """Mutual information (bits) between each feature and the label, descending."""
    y = train[target].to_numpy()
    mi = {f: mutual_info_score(train[f].to_numpy(), y) / np.log(2) for f in features}
    return pd.Series(mi).sort_values(ascending=False)
