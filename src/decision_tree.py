"""Binary CART decision tree with Gini impurity, in NumPy.

Split search
    For a node with samples S and a candidate feature f, the sorted distinct values
    v_1 < ... < v_k of f in S define k-1 candidate thresholds t_j = (v_j + v_{j+1}) / 2
    (midpoints, as in scikit-learn). Rows with x_f <= t_j go left. For the coded phishing
    features (-1/0/1) this gives at most two thresholds per feature: -0.5 separates
    {-1} from {0, 1} and 0.5 separates {-1, 0} from {1}.

    Class counts per distinct value are computed once with ``bincount``; cumulative sums
    then give the left/right counts for every threshold at once, so each feature costs
    O(|S| + k) instead of O(|S| k).

    Gini(S)   = 1 - Σ_c p_c²
    Weighted  = (|L| Gini(L) + |R| Gini(R)) / |S|
    The split with the lowest weighted Gini wins (= largest impurity decrease).

Tie-breaking (deterministic): among equally good splits the lowest feature index wins,
then the smallest threshold.

Stopping: node is pure, depth == max_depth, fewer than ``min_samples_split`` samples,
no threshold leaves ``min_samples_leaf`` on both sides, or the impurity decrease is
below ``min_impurity_decrease``. Zero-gain splits of impure nodes are allowed (as in
scikit-learn), because they can enable useful splits deeper down (e.g. XOR).

``max_features`` (random feature subsets, used by the forest): features are visited
in a random order; constant features are skipped without counting; search stops once
``max_features`` non-constant features have been evaluated *and* a valid split exists.

The tree is grown with an explicit stack rather than Python recursion, so deep trees
cannot hit the interpreter's recursion limit. It is stored as flat arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LEAF = -1


def gini(counts: np.ndarray) -> np.ndarray:
    """Gini impurity for class-count rows (works on 1-D or 2-D arrays)."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    return np.where(n[..., 0] > 0, 1.0 - (p ** 2).sum(axis=-1), 0.0)


def resolve_max_features(max_features: int | float | str | None, n_features: int) -> int:
    """Translate sklearn-style ``max_features`` into a feature count."""
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, float):
        if not 0 < max_features <= 1:
            raise ValueError("float max_features must be in (0, 1]")
        return max(1, int(max_features * n_features))
    if isinstance(max_features, (int, np.integer)) and 1 <= max_features <= n_features:
        return int(max_features)
    raise ValueError(f"invalid max_features: {max_features!r}")


def _check_X(X: np.ndarray, n_features: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, tree expects {n_features}")
    return X


class DecisionTree:
    """Gini decision tree for binary labels {0, 1}.

    Args:
        max_depth: Maximum depth (root = depth 0); ``None`` = unlimited.
        min_samples_split: Minimum samples a node needs to be considered for splitting.
        min_samples_leaf: Minimum samples in each child.
        max_features: Candidate features per split: None, int, float, "sqrt" or "log2".
        min_impurity_decrease: Minimum weighted impurity decrease (n_node/n_root · Δgini).
        seed: Seed (or ``np.random.Generator``) for feature subsampling.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        min_impurity_decrease: float = 0.0,
        seed: int | np.random.Generator | None = 42,
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2 or min_samples_leaf < 1:
            raise ValueError("min_samples_split >= 2 and min_samples_leaf >= 1 required")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.seed = seed
        self.n_features: int | None = None

    # ------------------------------------------------------------------ split search

    def _best_split(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray,
                    rng: np.random.Generator) -> tuple[int, float, float] | None:
        """Return (feature, threshold, weighted child gini) of the best valid split."""
        n = len(idx)
        yi = y[idx]
        k = self._k_features
        order = np.arange(self.n_features) if k == self.n_features else rng.permutation(self.n_features)
        best: tuple[int, float, float] | None = None
        evaluated = 0
        for f in order:
            if evaluated >= k and best is not None:
                break
            u, inv = np.unique(X[idx, f], return_inverse=True)
            if len(u) < 2:
                continue  # constant in this node: not counted towards max_features
            evaluated += 1
            cnt = np.bincount(inv * 2 + yi, minlength=2 * len(u)).reshape(-1, 2)
            left = np.cumsum(cnt, axis=0)[:-1]
            right = cnt.sum(axis=0) - left
            nl, nr = left.sum(axis=1), right.sum(axis=1)
            valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            if not valid.any():
                continue
            w = (nl * gini(left) + nr * gini(right)) / n
            w[~valid] = np.inf
            j = int(np.argmin(w))  # first minimum -> smallest threshold
            cand = (int(f), float((u[j] + u[j + 1]) / 2), float(w[j]))
            if best is None or cand[2] < best[2] - 1e-12 or (abs(cand[2] - best[2]) <= 1e-12 and cand[0] < best[0]):
                best = cand
        return best

    # ------------------------------------------------------------------ training

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Grow the tree on (X, y)."""
        X = _check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one label per row of X")
        if len(y) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must contain only 0 and 1")
        y = y.astype(int)
        self.n_features = X.shape[1]
        self._k_features = resolve_max_features(self.max_features, self.n_features)
        rng = self.seed if isinstance(self.seed, np.random.Generator) else np.random.default_rng(self.seed)
        max_depth = np.inf if self.max_depth is None else self.max_depth
        n_root = len(y)

        feature, threshold, left, right, value, n_samples, impurity, depth_arr = ([] for _ in range(8))
        importance = np.zeros(self.n_features)

        def new_node(idx: np.ndarray, depth: int) -> int:
            counts = np.bincount(y[idx], minlength=2)
            feature.append(LEAF); threshold.append(0.0); left.append(LEAF); right.append(LEAF)
            value.append(float(counts[1] / counts.sum())); n_samples.append(int(len(idx)))
            impurity.append(float(gini(counts))); depth_arr.append(depth)
            return len(feature) - 1

        root = new_node(np.arange(n_root), 0)
        stack = [(root, np.arange(n_root), 0)]
        while stack:
            node, idx, depth = stack.pop()
            n = len(idx)
            if (impurity[node] <= 0.0 or depth >= max_depth or n < self.min_samples_split
                    or n < 2 * self.min_samples_leaf):
                continue
            split = self._best_split(X, y, idx, rng)
            if split is None:
                continue
            f, t, w_child = split
            decrease = n / n_root * (impurity[node] - w_child)
            if decrease + 1e-12 < self.min_impurity_decrease:
                continue
            mask = X[idx, f] <= t
            li, ri = idx[mask], idx[~mask]
            feature[node], threshold[node] = f, t
            importance[f] += max(decrease, 0.0)
            left[node] = new_node(li, depth + 1)
            right[node] = new_node(ri, depth + 1)
            # Push right first so the left subtree is expanded first (deterministic node order).
            stack.append((right[node], ri, depth + 1))
            stack.append((left[node], li, depth + 1))

        self.feature_ = np.array(feature, dtype=int)
        self.threshold_ = np.array(threshold, dtype=float)
        self.left_ = np.array(left, dtype=int)
        self.right_ = np.array(right, dtype=int)
        self.value_ = np.array(value, dtype=float)
        self.n_samples_ = np.array(n_samples, dtype=int)
        self.impurity_ = np.array(impurity, dtype=float)
        self.node_depth_ = np.array(depth_arr, dtype=int)
        self.raw_importance_ = importance
        return self

    # ------------------------------------------------------------------ inference

    def _check_fitted(self) -> None:
        if not hasattr(self, "feature_"):
            raise RuntimeError("tree is not fitted")

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Index of the leaf each row lands in (vectorised level-by-level traversal)."""
        self._check_fitted()
        X = _check_X(X, self.n_features)
        node = np.zeros(len(X), dtype=int)
        active = self.feature_[node] != LEAF
        while active.any():
            rows = np.flatnonzero(active)
            nd = node[rows]
            go_left = X[rows, self.feature_[nd]] <= self.threshold_[nd]
            node[rows] = np.where(go_left, self.left_[nd], self.right_[nd])
            active[rows] = self.feature_[node[rows]] != LEAF
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Phishing probability = class-1 share of the training samples in the leaf."""
        leaves = self.apply(X)  # checks fitted state and input first
        return self.value_[leaves]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------ properties

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean-decrease-in-impurity importance, normalised to sum to 1."""
        self._check_fitted()
        total = self.raw_importance_.sum()
        return self.raw_importance_ / total if total > 0 else self.raw_importance_.copy()

    @property
    def n_nodes(self) -> int:
        self._check_fitted()
        return len(self.feature_)

    @property
    def n_leaves(self) -> int:
        self._check_fitted()
        return int((self.feature_ == LEAF).sum())

    @property
    def depth(self) -> int:
        self._check_fitted()
        return int(self.node_depth_.max())

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "params": {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                       "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features,
                       "min_impurity_decrease": self.min_impurity_decrease,
                       "seed": self.seed if isinstance(self.seed, (int, type(None))) else None},
            "n_features": self.n_features,
            "feature": self.feature_.tolist(),
            "threshold": self.threshold_.tolist(),
            "left": self.left_.tolist(),
            "right": self.right_.tolist(),
            "value": self.value_.tolist(),
            "n_samples": self.n_samples_.tolist(),
            "impurity": self.impurity_.tolist(),
            "depth": self.node_depth_.tolist(),
            "raw_importance": self.raw_importance_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionTree":
        t = cls(**d["params"])
        t.n_features = d["n_features"]
        t.feature_ = np.array(d["feature"], dtype=int)
        t.threshold_ = np.array(d["threshold"], dtype=float)
        t.left_ = np.array(d["left"], dtype=int)
        t.right_ = np.array(d["right"], dtype=int)
        t.value_ = np.array(d["value"], dtype=float)
        t.n_samples_ = np.array(d["n_samples"], dtype=int)
        t.impurity_ = np.array(d["impurity"], dtype=float)
        t.node_depth_ = np.array(d["depth"], dtype=int)
        t.raw_importance_ = np.array(d["raw_importance"], dtype=float)
        return t

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DecisionTree":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def export_text(self, feature_names: list[str] | None = None, max_depth: int = 3) -> str:
        """Readable outline of the top of the tree."""
        self._check_fitted()
        names = feature_names or [f"x{i}" for i in range(self.n_features)]
        lines: list[str] = []

        def walk(node: int, indent: int) -> None:
            pad = "|   " * indent
            if self.feature_[node] == LEAF or indent >= max_depth:
                lines.append(f"{pad}leaf: P(phishing)={self.value_[node]:.3f} (n={self.n_samples_[node]})")
                return
            f, t = names[self.feature_[node]], self.threshold_[node]
            lines.append(f"{pad}{f} <= {t:g}  (n={self.n_samples_[node]}, gini={self.impurity_[node]:.3f})")
            walk(self.left_[node], indent + 1)
            lines.append(f"{pad}{f} >  {t:g}")
            walk(self.right_[node], indent + 1)

        walk(0, 0)
        return "\n".join(lines)
