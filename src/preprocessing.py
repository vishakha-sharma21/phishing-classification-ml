"""Cleaning, label mapping and the stratified 70/15/15 split.

The duplicate policy was approved after Phase 1 (see reports/cleaning_log.json):
collapse exact feature+label duplicates, drop every feature vector observed with
both labels, then split. Encoding and baselines are added in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABEL_MAPPING, RANDOM_SEED, TARGET_COLUMN

TARGET = "phishing"  # project target column: 1 = phishing, 0 = legitimate


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return predictor columns in file order (excludes original and mapped targets)."""
    return [c for c in df.columns if c not in (TARGET_COLUMN, TARGET)]


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the original ``result`` (-1/+1) with ``phishing`` (1/0).

    Raises:
        ValueError: If labels outside the documented mapping are present.
    """
    unknown = set(df[TARGET_COLUMN].unique()) - set(LABEL_MAPPING)
    if unknown:
        raise ValueError(f"Unexpected target values: {sorted(unknown)}")
    out = df.drop(columns=TARGET_COLUMN)
    out[TARGET] = df[TARGET_COLUMN].map(LABEL_MAPPING).astype(int)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the approved duplicate policy to raw data and map labels.

    Returns a new frame with a fresh 0..n-1 index; row order follows first occurrence.
    """
    features = feature_columns(df)
    unique = df.drop_duplicates(keep="first")
    labels_per_vector = unique.groupby(features, sort=False)[TARGET_COLUMN].transform("nunique")
    kept = unique[labels_per_vector.values == 1]
    return map_labels(kept).reset_index(drop=True)


@dataclass(frozen=True)
class Split:
    """Row indices (into the cleaned frame) of each partition."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_indices(
    y: pd.Series | np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Split:
    """Stratified train/validation/test split of positions ``0..len(y)-1``.

    Two-stage: carve off the test set, then split the remainder into train and
    validation so the proportions are ``train_size / val_size / rest``.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    # Rounded: 1 - 0.70 - 0.15 is 0.15000000000000002, which sklearn would ceil up a row.
    test_size = round(1.0 - train_size - val_size, 10)
    if not 0 < test_size < 1:
        raise ValueError("train_size + val_size must be in (0, 1)")
    rest, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    train, val = train_test_split(
        rest, test_size=val_size / (train_size + val_size), stratify=y[rest], random_state=seed
    )
    return Split(np.sort(train), np.sort(val), np.sort(test))
