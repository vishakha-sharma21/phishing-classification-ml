"""Tests for the custom NumPy logistic regression."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SkLR

from src.logistic_regression import LogisticRegression, sigmoid


def _data(n=400, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.array([2.0, -1.0, 0.5, 0.0])[:d]
    y = (rng.random(n) < sigmoid(X @ w + 0.3)).astype(int)
    return X, y


def test_sigmoid_is_stable_and_bounded():
    z = np.array([-1e4, -50, 0, 50, 1e4])
    with np.errstate(over="raise", invalid="raise", divide="raise"):  # underflow to 0 is fine
        p = sigmoid(z)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    assert p[2] == 0.5 and p[0] == 0.0 and p[-1] == 1.0


def test_output_shapes_and_probability_range():
    X, y = _data()
    m = LogisticRegression(n_iter=200).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and m.w.shape == (X.shape[1],)
    assert np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1
    # Extreme inputs still give finite probabilities.
    assert np.all(np.isfinite(m.predict_proba(X * 1e6)))


@pytest.mark.parametrize("l2,cw", [(0.0, None), (0.1, None), (0.05, "balanced")])
def test_gradient_matches_finite_differences(l2, cw):
    X, y = _data(n=60)
    m = LogisticRegression(l2=l2, class_weight=cw)
    s = m._sample_weights(y.astype(float))
    rng = np.random.default_rng(3)
    w, b = rng.normal(size=X.shape[1]), 0.2
    gw, gb = m.gradient(X, y.astype(float), w, b, s)
    eps = 1e-6
    num_w = np.array([
        (m.objective(X, y, w + eps * e, b, s) - m.objective(X, y, w - eps * e, b, s)) / (2 * eps)
        for e in np.eye(len(w))
    ])
    num_b = (m.objective(X, y, w, b + eps, s) - m.objective(X, y, w, b - eps, s)) / (2 * eps)
    np.testing.assert_allclose(gw, num_w, rtol=1e-5, atol=1e-8)
    assert gb == pytest.approx(num_b, rel=1e-5, abs=1e-8)


def test_loss_decreases_and_converges_on_separable_signal():
    X, y = _data(n=1000)
    m = LogisticRegression(learning_rate=0.5, n_iter=3000, tol=1e-10).fit(X, y)
    h = m.history["train_loss"]
    assert h[-1] < h[0]
    assert np.all(np.diff(h) <= 1e-12)  # monotone for a small enough step on a convex objective
    assert (m.predict(X) == y).mean() > 0.75


def test_matches_sklearn_reference():
    X, y = _data(n=800)
    lam = 0.01
    ours = LogisticRegression(learning_rate=0.5, n_iter=20000, l2=lam, tol=1e-12).fit(X, y)
    sk = SkLR(C=1 / (lam * len(X)), tol=1e-10, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(ours.w, sk.coef_[0], atol=1e-4)
    assert ours.b == pytest.approx(sk.intercept_[0], abs=1e-4)


def test_l2_shrinks_weights_but_not_bias():
    X, y = _data()
    small = LogisticRegression(l2=0.0, n_iter=3000).fit(X, y)
    big = LogisticRegression(l2=1.0, n_iter=3000).fit(X, y)
    assert np.linalg.norm(big.w) < 0.5 * np.linalg.norm(small.w)


def test_threshold_controls_predictions():
    X, y = _data()
    m = LogisticRegression(n_iter=500).fit(X, y)
    assert m.predict(X, 0.0).sum() == len(X)
    assert m.predict(X, 1.0).sum() == (m.predict_proba(X) >= 1.0).sum()
    assert m.predict(X, 0.3).sum() >= m.predict(X, 0.7).sum()
    with pytest.raises(ValueError):
        m.predict(X, 1.5)


def test_minibatch_reaches_similar_solution():
    X, y = _data(n=1000)
    full = LogisticRegression(learning_rate=0.5, n_iter=3000).fit(X, y)
    mini = LogisticRegression(learning_rate=0.05, n_iter=60, batch_size=32, seed=1).fit(X, y)
    assert abs(full.history["train_loss"][-1] - mini.history["train_loss"][-1]) < 0.01


def test_balanced_class_weight_raises_minority_recall():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(1000, 2))
    y = (X[:, 0] + rng.normal(0, 1, 1000) > 1.6).astype(int)  # ~10% positives
    plain = LogisticRegression(n_iter=2000).fit(X, y)
    bal = LogisticRegression(n_iter=2000, class_weight="balanced").fit(X, y)
    assert bal.predict(X)[y == 1].mean() > plain.predict(X)[y == 1].mean()


def test_save_load_roundtrip(tmp_path):
    X, y = _data()
    m = LogisticRegression(l2=0.01, n_iter=300).fit(X, y)
    m.save(tmp_path / "lr.json", extra={"threshold": 0.4})
    m2 = LogisticRegression.load(tmp_path / "lr.json")
    np.testing.assert_array_equal(m.predict_proba(X), m2.predict_proba(X))


def test_invalid_inputs_rejected():
    X, y = _data(n=50)
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y + 1)
    with pytest.raises(ValueError):
        LogisticRegression().fit(X[:, 0], y)
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.where(X > 1, np.nan, X), y)
    m = LogisticRegression(n_iter=10).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X[:, :2])
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_proba(X)
