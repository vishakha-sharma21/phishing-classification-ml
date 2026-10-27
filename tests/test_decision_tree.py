"""Tests for the custom Gini decision tree."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.decision_tree import DecisionTree, gini, resolve_max_features


def test_gini_values():
    assert gini(np.array([5, 5])) == pytest.approx(0.5)
    assert gini(np.array([10, 0])) == 0.0
    assert gini(np.array([0, 0])) == 0.0  # empty node is safe
    np.testing.assert_allclose(gini(np.array([[3, 1], [2, 2]])), [0.375, 0.5])


def test_perfectly_separable_one_split():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTree().fit(X, y)
    assert t.n_nodes == 3 and t.depth == 1
    assert t.threshold_[0] == 1.5  # midpoint between distinct values
    assert t.predict(X).tolist() == y.tolist()


def test_ternary_codes_get_midpoint_thresholds():
    X = np.array([[-1], [-1], [0], [0], [1], [1]], dtype=float)
    y = np.array([1, 1, 1, 0, 0, 0])
    t = DecisionTree().fit(X, y)
    assert set(t.threshold_[t.feature_ >= 0]) <= {-0.5, 0.5}


def test_pure_node_is_a_leaf():
    X = np.random.default_rng(0).normal(size=(20, 3))
    t = DecisionTree().fit(X, np.ones(20, dtype=int))
    assert t.n_nodes == 1 and t.predict_proba(X).tolist() == [1.0] * 20


def test_constant_features_never_split():
    rng = np.random.default_rng(1)
    X = np.c_[np.ones(100), rng.normal(size=100), np.zeros(100)]
    y = (X[:, 1] > 0).astype(int)
    t = DecisionTree(max_features=1, seed=3).fit(X, y)
    used = set(t.feature_[t.feature_ >= 0])
    assert used == {1}
    # All-constant data: no valid split, single leaf, no crash.
    t2 = DecisionTree().fit(np.ones((10, 2)), np.array([0, 1] * 5))
    assert t2.n_nodes == 1 and t2.value_[0] == 0.5


def test_min_samples_leaf_and_split_respected():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + rng.normal(0, 0.5, 300) > 0).astype(int)
    t = DecisionTree(min_samples_leaf=15).fit(X, y)
    assert t.n_samples_[t.feature_ < 0].min() >= 15
    t2 = DecisionTree(min_samples_split=100).fit(X, y)
    internal = t2.feature_ >= 0
    assert t2.n_samples_[internal].min() >= 100


def test_max_depth_respected_and_unlimited_tree_fits_training_data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5))
    y = rng.integers(0, 2, 400)  # pure noise
    for d in (0, 1, 3, 6):
        assert DecisionTree(max_depth=d).fit(X, y).depth <= d
    full = DecisionTree().fit(X, y)
    assert (full.predict(X) == y).mean() == 1.0  # memorises distinct points


def test_probabilities_are_leaf_frequencies():
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0]])
    y = np.array([1, 0, 0, 1, 1])
    t = DecisionTree().fit(X, y)
    np.testing.assert_allclose(t.predict_proba(np.array([[0.0], [1.0]])), [1 / 3, 1.0])
    assert t.predict(np.array([[0.0]]), threshold=0.3).tolist() == [1]


def test_xor_needs_zero_gain_first_split():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    y = np.array([0, 1, 1, 0] * 5)
    assert (DecisionTree().fit(X, y).predict(X) == y).all()


def test_feature_importance_sums_to_one_and_targets_signal():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 4))
    y = (X[:, 2] > 0.3).astype(int)
    imp = DecisionTree(max_depth=3).fit(X, y).feature_importances_
    assert imp.sum() == pytest.approx(1.0) and np.argmax(imp) == 2


def test_serialization_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.integers(-1, 2, size=(200, 6)).astype(float)
    y = rng.integers(0, 2, 200)
    t = DecisionTree(max_depth=5, max_features="sqrt", seed=7).fit(X, y)
    t.save(tmp_path / "tree.json")
    t2 = DecisionTree.load(tmp_path / "tree.json")
    np.testing.assert_array_equal(t.predict_proba(X), t2.predict_proba(X))
    np.testing.assert_array_equal(t.feature_importances_, t2.feature_importances_)


def test_deterministic_given_seed():
    rng = np.random.default_rng(6)
    X = rng.integers(-1, 2, size=(300, 10)).astype(float)
    y = rng.integers(0, 2, 300)
    a = DecisionTree(max_features=3, seed=11).fit(X, y)
    b = DecisionTree(max_features=3, seed=11).fit(X, y)
    np.testing.assert_array_equal(a.feature_, b.feature_)


def _weighted_gini(X, y, idx, f, t):
    m = X[idx, f] <= t
    l, r = idx[m], idx[~m]
    return (len(l) * gini(np.bincount(y[l], minlength=2)) + len(r) * gini(np.bincount(y[r], minlength=2))) / len(idx)


@pytest.mark.parametrize("params", [{}, {"max_depth": 4}, {"min_samples_leaf": 10}, {"max_depth": 6, "min_samples_leaf": 3}])
def test_matches_sklearn_up_to_tie_breaking(params):
    """Walk our tree and sklearn's in parallel. Wherever the chosen splits differ, the two
    candidate splits must have exactly the same weighted Gini (a tie that sklearn breaks with
    a random feature order and we break by lowest feature index). Thresholds are compared
    with a tolerance because sklearn stores X as float32."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 5))
    y = ((X[:, 0] * X[:, 1] > 0) ^ (rng.random(300) < 0.1)).astype(int)
    ours = DecisionTree(**params).fit(X, y)
    sk = DecisionTreeClassifier(random_state=0, **params).fit(X, y)
    tr = sk.tree_
    stack, compared, ties = [(0, 0, np.arange(len(y)))], 0, 0
    while stack:
        a, b, idx = stack.pop()
        fa, fb = ours.feature_[a], tr.feature[b]
        assert (fa < 0) == (fb < 0), "one tree split a node the other kept as a leaf"
        if fa < 0:
            assert ours.value_[a] == pytest.approx(tr.value[b][0][1] / tr.value[b][0].sum())
            continue
        ta, tb = ours.threshold_[a], tr.threshold[b]
        if fa != fb or abs(ta - tb) > 1e-5:
            assert _weighted_gini(X, y, idx, fa, ta) == pytest.approx(_weighted_gini(X, y, idx, fb, tb), abs=1e-12)
            ties += 1  # subtrees below a differently-broken tie may legitimately differ
            continue
        compared += 1
        m = X[idx, fa] <= ta
        stack += [(ours.left_[a], tr.children_left[b], idx[m]), (ours.right_[a], tr.children_right[b], idx[~m])]
    assert compared >= 3
    if ties == 0:
        np.testing.assert_array_equal(ours.predict(X), sk.predict(X))
    if not params:  # unconstrained trees both fit the training data exactly
        assert (ours.predict(X) == y).all() and (sk.predict(X) == y).all()


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        DecisionTree().fit(np.ones((3, 2)), np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        DecisionTree().fit(np.ones(3), np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        DecisionTree(min_samples_leaf=0)
    t = DecisionTree().fit(np.ones((4, 2)), np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError):
        t.predict(np.ones((2, 3)))
    with pytest.raises(RuntimeError):
        DecisionTree().predict(np.ones((2, 2)))
    with pytest.raises(ValueError):
        resolve_max_features(1.5, 10)
    assert resolve_max_features("sqrt", 30) == 5 and resolve_max_features(None, 30) == 30


@pytest.mark.parametrize("params", [{}, {"max_depth": 5, "min_samples_leaf": 3}, {"max_features": "sqrt"},
                                    {"max_features": 2, "min_samples_leaf": 4}])
def test_fast_path_grows_identical_tree_to_general_path(params, monkeypatch):
    import src.decision_tree as dt

    rng = np.random.default_rng(8)
    X = rng.integers(-1, 2, size=(600, 12)).astype(float)
    X[:, 3] = rng.integers(0, 2, 600)          # a {0,1} feature
    X[rng.random(600) < 0.3, 5] = -1           # a feature with an often-missing middle level
    y = ((X[:, 0] - X[:, 1] + 0.5 * X[:, 5] + rng.normal(0, 1, 600)) > 0).astype(int)
    fast = DecisionTree(seed=13, **params).fit(X, y)
    monkeypatch.setattr(dt, "MAX_CODES_FAST", 0)
    slow = DecisionTree(seed=13, **params).fit(X, y)
    assert not slow._fast and fast._fast
    for a in ("feature_", "threshold_", "left_", "right_", "value_", "n_samples_"):
        np.testing.assert_array_equal(getattr(fast, a), getattr(slow, a))
