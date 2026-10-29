"""Decision-threshold analysis and selection (validation data only)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.evaluation import compute_metrics


def threshold_curve(y: np.ndarray, proba: np.ndarray, grid: np.ndarray | None = None) -> dict[str, list]:
    """Precision, recall and F1 across thresholds."""
    grid = np.round(np.linspace(0.05, 0.95, 91), 3) if grid is None else grid
    rows = [compute_metrics(y, proba, t) for t in grid]
    return {"threshold": grid.tolist(), **{k: [r[k] for r in rows] for k in ("precision", "recall", "f1", "accuracy")}}


def tune_threshold(
    y: np.ndarray, proba: np.ndarray, objective: Callable[[dict], float] | str = "f1"
) -> float:
    """Validation threshold maximising ``objective``; ties go to the threshold closest to 0.5."""
    grid = np.round(np.linspace(0.05, 0.95, 91), 3)
    score = (lambda m: m[objective]) if isinstance(objective, str) else objective
    vals = np.array([score(compute_metrics(y, proba, t)) for t in grid])
    best = np.flatnonzero(np.isclose(vals, vals.max()))
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])


ZONES = ("Safe", "Suspicious", "Likely phishing")


def zone_thresholds(y: np.ndarray, proba: np.ndarray, target: float = 0.98) -> dict[str, float]:
    """Two cut-offs for a three-level verdict, chosen on validation data.

    * ``low``  — the largest threshold that still flags at least ``target`` of phishing sites
      (recall ≥ target), so at most 1 − target of phishing sites end up in "Safe".
    * ``high`` — the smallest threshold at which at least ``target`` of flagged sites are
      phishing (precision ≥ target), so "Likely phishing" is rarely a false alarm.

    Sites between the two are "Suspicious". If the constraints overlap, ``high`` is raised
    to ``low`` and the middle zone is empty.
    """
    y, proba = np.asarray(y).astype(int), np.asarray(proba, dtype=float)
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rec = np.array([compute_metrics(y, proba, t)["recall"] for t in grid])
    prec = np.array([compute_metrics(y, proba, t)["precision"] if (proba >= t).any() else 0.0 for t in grid])
    ok_low, ok_high = grid[rec >= target], grid[prec >= target]
    if len(ok_low) == 0 or len(ok_high) == 0:
        raise ValueError(f"no threshold reaches the {target:.0%} target on this data")
    low = float(ok_low.max())
    return {"low": low, "high": float(max(ok_high.min(), low)), "target": target}


def assign_zone(proba: np.ndarray, low: float, high: float) -> np.ndarray:
    """Zone index per row: 0 = Safe (p < low), 1 = Suspicious, 2 = Likely phishing (p ≥ high)."""
    p = np.asarray(proba, dtype=float)
    return np.where(p >= high, 2, np.where(p >= low, 1, 0))


def zone_table(y: np.ndarray, proba: np.ndarray, low: float, high: float) -> dict[str, dict[str, float]]:
    """How many rows fall in each zone and the observed phishing rate there."""
    z, y = assign_zone(proba, low, high), np.asarray(y)
    return {name: {"rows": int((z == i).sum()), "share": float((z == i).mean()),
                   "phishing": int(y[z == i].sum()),
                   "phishing_rate": float(y[z == i].mean()) if (z == i).any() else float("nan")}
            for i, name in enumerate(ZONES)}
