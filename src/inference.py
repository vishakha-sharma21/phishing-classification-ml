"""Load saved artifacts and score pre-extracted feature rows (no training, no network).

Used by app.py; kept separate so it can be unit-tested without Streamlit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.logistic_regression import LogisticRegression
from src.preprocessing import FeatureEncoder, validate_features
from src.random_forest import RandomForest


@dataclass
class Artifacts:
    """Everything needed to score new rows, exactly as saved by ``python -m src.train``."""

    feature_order: list[str]
    allowed_values: dict[str, list[int]]
    lr: LogisticRegression
    lr_encoder: FeatureEncoder
    lr_threshold: float
    rf: RandomForest
    rf_encoder: FeatureEncoder
    rf_threshold: float
    results: dict


def load_artifacts(models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR) -> Artifacts:
    """Read preprocessing config, both models and the results file.

    Raises:
        FileNotFoundError: if training has not been run.
        ValueError: if a model's input names do not match the preprocessing config.
    """
    pre = json.loads((models_dir / "preprocessing.json").read_text(encoding="utf-8"))
    lr_doc = json.loads((models_dir / "logistic_regression.json").read_text(encoding="utf-8"))
    rf_doc = json.loads((models_dir / "random_forest.json").read_text(encoding="utf-8"))
    results_path = reports_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}

    lr_enc = FeatureEncoder.from_dict(pre["encoders"][lr_doc["encoding"]])
    rf_enc = FeatureEncoder.from_dict(pre["encoders"][rf_doc["encoding"]])
    for doc, e in ((lr_doc, lr_enc), (rf_doc, rf_enc)):
        if doc["input_names"] != e.output_names:
            raise ValueError("saved model inputs do not match the preprocessing configuration")
    return Artifacts(
        feature_order=pre["feature_order"],
        allowed_values={k: [int(v) for v in vs] for k, vs in pre["allowed_values"].items()},
        lr=LogisticRegression.from_dict(lr_doc), lr_encoder=lr_enc, lr_threshold=float(lr_doc["threshold"]),
        rf=RandomForest.from_dict(rf_doc), rf_encoder=rf_enc, rf_threshold=float(rf_doc["threshold"]),
        results=results,
    )


def prepare_input(df: pd.DataFrame, art: Artifacts) -> tuple[pd.DataFrame | None, list[str]]:
    """Validate a user frame; return (clean integer frame in training order, problems)."""
    problems = validate_features(df, art.allowed_values)
    if len(df) == 0:
        problems.append("no rows supplied")
    if problems:
        return None, problems
    clean = df[art.feature_order].apply(pd.to_numeric).astype(int)
    return clean, []


def predict(df: pd.DataFrame, art: Artifacts) -> pd.DataFrame:
    """Phishing probability and class from both models for each validated row."""
    clean, problems = prepare_input(df, art)
    if problems:
        raise ValueError("; ".join(problems))
    p_lr = art.lr.predict_proba(art.lr_encoder.transform(clean))
    p_rf = art.rf.predict_proba(art.rf_encoder.transform(clean))
    label = lambda p, t: np.where(p >= t, "phishing", "legitimate")  # noqa: E731
    return pd.DataFrame({
        "lr_phishing_probability": p_lr.round(4), "lr_prediction": label(p_lr, art.lr_threshold),
        "rf_phishing_probability": p_rf.round(4), "rf_prediction": label(p_rf, art.rf_threshold),
    }, index=df.index)
