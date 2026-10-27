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
