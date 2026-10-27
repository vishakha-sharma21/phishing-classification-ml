"""Cleaning, label mapping, the stratified 70/15/15 split and feature encoding.

The duplicate policy was approved after Phase 1 (see reports/cleaning_log.json):
collapse exact feature+label duplicates, drop every feature vector observed with
both labels, then split.

Encodings (fitted on the training partition only):
* ``raw``    — the original codes (-1/0/1) as numbers. Default for trees.
* ``onehot`` — one indicator column per non-reference level; the reference level of
  every feature is its highest code (1, the "legitimate-like" level), so each
  coefficient is the change in log-odds relative to that level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def validate_features(df: pd.DataFrame, levels: dict[str, list[int]]) -> list[str]:
    """Return a list of human-readable problems (empty if the frame is valid).

    Checks that every expected feature is present, has no missing values, and only
    contains integer codes seen during training.
    """
    problems = []
    missing_cols = [f for f in levels if f not in df.columns]
    if missing_cols:
        problems.append(f"missing columns: {missing_cols}")
    for f, allowed in levels.items():
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        n_nan = int(col.isna().sum())
        if n_nan:
            problems.append(f"{f}: {n_nan} missing or non-numeric value(s)")
        bad = sorted(set(col.dropna().unique()) - set(allowed))
        if bad:
            problems.append(f"{f}: invalid value(s) {bad}; allowed {allowed}")
    return problems


class FeatureEncoder:
    """Maps a frame of the 30 coded features to a numeric matrix in a fixed order."""

    def __init__(self, kind: str = "raw") -> None:
        if kind not in ("raw", "onehot"):
            raise ValueError("kind must be 'raw' or 'onehot'")
        self.kind = kind
        self.features: list[str] = []
        self.levels: dict[str, list[int]] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """Learn feature order and the levels present in the (training) frame."""
        self.features = feature_columns(df)
        self.levels = {f: sorted(int(v) for v in df[f].unique()) for f in self.features}
        return self

    @property
    def output_names(self) -> list[str]:
        """Column names of the encoded matrix, in order."""
        if self.kind == "raw":
            return list(self.features)
        return [f"{f}={v}" for f in self.features for v in self.levels[f][:-1]]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Encode ``df``; raises ``ValueError`` on missing columns or unseen codes."""
        if not self.features:
            raise RuntimeError("encoder is not fitted")
        problems = validate_features(df, self.levels)
        if problems:
            raise ValueError("invalid input: " + "; ".join(problems))
        X = df[self.features].to_numpy(dtype=float)
        if self.kind == "raw":
            return X
        cols = [(X[:, j] == v) for j, f in enumerate(self.features) for v in self.levels[f][:-1]]
        return np.column_stack(cols).astype(float)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "features": self.features,
                "levels": {f: v for f, v in self.levels.items()}, "output_names": self.output_names}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureEncoder":
        enc = cls(d["kind"])
        enc.features = list(d["features"])
        enc.levels = {f: [int(x) for x in v] for f, v in d["levels"].items()}
        return enc


@dataclass
class PreparedData:
    """Everything downstream code needs: cleaned frame, split and arrays."""

    cleaned: pd.DataFrame
    split: Split
    features: list[str]

    def part(self, name: str) -> pd.DataFrame:
        return self.cleaned.iloc[getattr(self.split, name)]

    def y(self, name: str) -> np.ndarray:
        return self.part(name)[TARGET].to_numpy()


def prepare(raw: pd.DataFrame) -> PreparedData:
    """Clean the raw frame and create the approved split."""
    cleaned = clean(raw)
    return PreparedData(cleaned, split_indices(cleaned[TARGET]), feature_columns(cleaned))


def save_split(data: PreparedData, path: Path) -> None:
    """Persist split indices (positions in the cleaned frame) for reproducibility."""
    payload = {
        "seed": RANDOM_SEED,
        "proportions": [0.70, 0.15, 0.15],
        "stratified_by": TARGET,
        "index_refers_to": "row position in data/processed/cleaned.csv",
        **{k: getattr(data.split, k).tolist() for k in ("train", "val", "test")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
