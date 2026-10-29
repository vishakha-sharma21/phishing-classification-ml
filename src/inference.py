"""Load the app's saved Logistic Regression and score pre-extracted feature rows (no training, no network).

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


# ------------------------------------------------------------ quick (8-input) model

VERDICTS = ("Safe", "Suspicious", "Likely phishing")


@dataclass
class QuickArtifacts:
    """The app's 8-input Logistic Regression, as saved by ``python -m src.train --stage quick``."""

    features: list[str]
    allowed_values: dict[str, list[int]]
    lr: LogisticRegression
    lr_encoder: FeatureEncoder
    lr_threshold: float
    lr_zones: dict[str, float]
    lr_mean: np.ndarray
    lr_baseline_logit: float
    zone_rule: str
    results: dict


def load_quick_artifacts(models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR) -> QuickArtifacts:
    """Read the quick Logistic Regression and the results file.

    Raises:
        FileNotFoundError: if the quick stage has not been run.
        ValueError: if the saved encoder does not match the weights.
    """
    lr_doc = json.loads((models_dir / "quick_logistic_regression.json").read_text(encoding="utf-8"))
    results_path = reports_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    lr_enc = FeatureEncoder.from_dict(lr_doc["encoder"])
    if lr_enc.features != lr_doc["features"] or len(lr_enc.output_names) != len(lr_doc["weights"]):
        raise ValueError("quick LR encoder does not match its weights")
    return QuickArtifacts(
        features=lr_doc["features"], allowed_values=lr_enc.levels,
        lr=LogisticRegression.from_dict(lr_doc), lr_encoder=lr_enc, lr_threshold=float(lr_doc["threshold"]),
        lr_zones=lr_doc["zones"], lr_mean=np.asarray(lr_doc["contribution_mean"], dtype=float),
        lr_baseline_logit=float(lr_doc["baseline_logit"]), zone_rule=lr_doc["zone_rule"], results=results,
    )


def verdict_for(p: float | np.ndarray, zones: dict[str, float]) -> np.ndarray:
    """Three-level verdict: Safe (p < low), Suspicious, Likely phishing (p ≥ high)."""
    p = np.asarray(p, dtype=float)
    return np.array(VERDICTS)[np.where(p >= zones["high"], 2, np.where(p >= zones["low"], 1, 0))]


def predict_quick(df: pd.DataFrame, art: QuickArtifacts) -> pd.DataFrame:
    """Phishing probability and three-level verdict from the quick Logistic Regression.

    Raises:
        ValueError: if any of the 8 inputs is missing or has an invalid code.
    """
    problems = validate_features(df, art.allowed_values)
    if len(df) == 0:
        problems.append("no rows supplied")
    if problems:
        raise ValueError("; ".join(problems))
    clean = df[art.features].apply(pd.to_numeric).astype(int)
    p = art.lr.predict_proba(art.lr_encoder.transform(clean))
    return pd.DataFrame({"phishing_probability": p.round(4), "verdict": verdict_for(p, art.lr_zones)},
                        index=df.index)


def lr_contributions(row: dict[str, int], art: QuickArtifacts) -> pd.Series:
    """Each answer's push on the Logistic Regression log-odds, sorted by size.

    contribution_f = Σ_j w_j (x_j − mean_j) over the encoded columns j of feature f, where
    mean_j is the training average. Positive pushes towards phishing, negative towards
    legitimate, and the contributions sum to logit(p) − baseline_logit (an average training site).
    """
    x = art.lr_encoder.transform(pd.DataFrame([row]))[0]
    per_col = art.lr.w * (x - art.lr_mean)
    owner = [name.split("=")[0] for name in art.lr_encoder.output_names]
    contrib = pd.Series(per_col).groupby(owner).sum().reindex(art.features).fillna(0.0)
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index)
