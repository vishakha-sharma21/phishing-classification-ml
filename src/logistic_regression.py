r"""Binary logistic regression trained by (mini-)batch gradient descent, in NumPy.

Model
    z_i = w·x_i + b,            p_i = σ(z_i) = 1 / (1 + e^{-z_i})

Objective (s_i = per-sample weight, 1 unless class weighting is used)
    J(w, b) = Σ_i s_i ℓ_i / Σ_i s_i  +  (λ/2) ||w||²
    ℓ_i     = -[y_i log p_i + (1 - y_i) log(1 - p_i)]  =  log(1 + e^{z_i}) - y_i z_i

Gradients (the bias is not regularised)
    ∂J/∂w = Xᵀ (s ⊙ (p - y)) / Σ s  +  λ w
    ∂J/∂b = Σ_i s_i (p_i - y_i) / Σ s

Numerical safeguards
    * σ is evaluated piece-wise so ``exp`` never receives a large positive argument.
    * ℓ_i uses ``logaddexp(0, z) - y z``, which is exact for any finite z and never
      takes ``log(0)``.

Relation to scikit-learn: sklearn minimises C Σ ℓ_i + ½||w||², which has the same
minimiser as J when C = 1 / (λ n) and all s_i = 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _check_X(X: np.ndarray, n_features: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, model expects {n_features}")
    return X


def _check_y(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y must be 1-D with one label per row of X")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    return y.astype(float)


class LogisticRegression:
    """Logistic regression with optional L2 penalty, class weights and mini-batches.

    Args:
        learning_rate: Gradient-descent step size.
        n_iter: Maximum number of iterations (batch GD) or epochs (mini-batch).
        l2: Penalty strength λ on the weights (bias excluded).
        batch_size: ``None`` for full-batch gradient descent, else mini-batch size.
        class_weight: ``None``, ``"balanced"`` (s_i = n / (2 n_{y_i})) or ``{0: w0, 1: w1}``.
        tol: Stop when the full-batch objective improves by less than ``tol``.
        seed: Seed for mini-batch shuffling.
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        n_iter: int = 5000,
        l2: float = 0.0,
        batch_size: int | None = None,
        class_weight: str | dict | None = None,
        tol: float = 1e-8,
        seed: int = 42,
    ) -> None:
        if learning_rate <= 0 or n_iter < 1 or l2 < 0:
            raise ValueError("learning_rate and n_iter must be positive, l2 non-negative")
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.tol = tol
        self.seed = seed
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.n_iter_run: int = 0
        self.converged: bool = False

    # ------------------------------------------------------------- core maths

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(y)
        if self.class_weight == "balanced":
            n, n1 = len(y), y.sum()
            cw = {0: n / (2 * (n - n1)), 1: n / (2 * n1)}
        else:
            cw = {int(k): float(v) for k, v in self.class_weight.items()}
        return np.where(y == 1, cw[1], cw[0])

    def objective(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                  s: np.ndarray | None = None) -> float:
        """Regularised weighted binary cross-entropy J(w, b)."""
        z = X @ w + b
        s = np.ones_like(y) if s is None else s
        bce = np.logaddexp(0.0, z) - y * z
        return float(np.dot(s, bce) / s.sum() + 0.5 * self.l2 * np.dot(w, w))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                 s: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Analytic gradient (∂J/∂w, ∂J/∂b)."""
        s = np.ones_like(y) if s is None else s
        r = s * (sigmoid(X @ w + b) - y) / s.sum()
        return X.T @ r + self.l2 * w, float(r.sum())

    # ------------------------------------------------------------- training

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "LogisticRegression":
        """Fit by gradient descent from zero initialisation.

        The objective is convex, so zero initialisation is safe and deterministic.
        If validation data is given, its (unregularised) BCE is recorded each iteration.
        """
        X = _check_X(X)
        y = _check_y(y, len(X))
        if len(np.unique(y)) < 2:
            raise ValueError("training data must contain both classes")
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = _check_X(X_val, X.shape[1])
            y_val = _check_y(y_val, len(X_val))
        s = self._sample_weights(y)
        w, b = np.zeros(X.shape[1]), 0.0
        rng = np.random.default_rng(self.seed)
        self.history = {"train_loss": [], "val_loss": []}
        self.converged = False
        s_sum = s.sum()
        prev = np.inf
        for it in range(1, self.n_iter + 1):
            if self.batch_size is None:
                # One pass: logits -> loss at the current parameters -> gradient -> step.
                z = X @ w + b
                cur = float(np.dot(s, np.logaddexp(0.0, z) - y * z) / s_sum + 0.5 * self.l2 * np.dot(w, w))
                r = s * (sigmoid(z) - y) / s_sum
                w, b = w - self.learning_rate * (X.T @ r + self.l2 * w), b - self.learning_rate * float(r.sum())
            else:
                order = rng.permutation(len(X))
                for start in range(0, len(X), self.batch_size):
                    idx = order[start:start + self.batch_size]
                    gw, gb = self.gradient(X[idx], y[idx], w, b, s[idx])
                    w, b = w - self.learning_rate * gw, b - self.learning_rate * gb
                cur = self.objective(X, y, w, b, s)
            if not np.isfinite(cur):
                raise FloatingPointError("objective diverged; reduce learning_rate")
            self.history["train_loss"].append(cur)
            if has_val:
                zv = X_val @ w + b
                self.history["val_loss"].append(float(np.mean(np.logaddexp(0.0, zv) - y_val * zv)))
            self.n_iter_run = it
            if self.batch_size is None and abs(prev - cur) < self.tol:
                self.converged = True
                break
            prev = cur
        self.w, self.b = w, b
        return self

    @staticmethod
    def smoothness_constant(X: np.ndarray, l2: float = 0.0) -> float:
        """L = ¼ λ_max(X̃ᵀX̃ / n) + λ for X̃ = [X, 1]: gradient descent is guaranteed to
        decrease J for any learning rate below 2 / L (unit sample weights)."""
        Xb = np.c_[np.asarray(X, dtype=float), np.ones(len(X))]
        return float(0.25 * np.linalg.eigvalsh(Xb.T @ Xb / len(X))[-1] + l2)

    #  inference

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Logits z = Xw + b."""
        if self.w is None:
            raise RuntimeError("model is not fitted")
        return _check_X(X, len(self.w)) @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Phishing probability for each row (1-D array)."""
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Class labels: 1 when probability >= threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(X) >= threshold).astype(int)

    #  persistence

    def to_dict(self) -> dict[str, Any]:
        if self.w is None:
            raise RuntimeError("model is not fitted")
        return {
            "model": "LogisticRegression (custom, NumPy)",
            "params": {"learning_rate": self.learning_rate, "n_iter": self.n_iter, "l2": self.l2,
                       "batch_size": self.batch_size, "class_weight": self.class_weight,
                       "tol": self.tol, "seed": self.seed},
            "weights": self.w.tolist(),
            "bias": self.b,
            "n_iter_run": self.n_iter_run,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogisticRegression":
        m = cls(**d["params"])
        m.w = np.asarray(d["weights"], dtype=float)
        m.b = float(d["bias"])
        m.n_iter_run = d.get("n_iter_run", 0)
        m.converged = d.get("converged", False)
        return m

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        """Write the model (plus e.g. feature names and threshold) as JSON."""
        Path(path).write_text(json.dumps({**self.to_dict(), **(extra or {})}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticRegression":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
