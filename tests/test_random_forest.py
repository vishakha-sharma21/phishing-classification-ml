"""Tests for the custom Random Forest."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(-1, 2, size=(n, 10)).astype(float)
    logit = 1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2] * X[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_shapes_range_and_tree_count():
    X, y = _data()
    rf = RandomForest(n_estimators=15, seed=1).fit(X, y)
    p = rf.predict_proba(X)
    assert len(rf.trees) == 15 and p.shape == (len(X),)
    assert p.min() >= 0 and p.max() <= 1
    assert rf.feature_importances_.sum() == pytest.approx(1.0)


def test_probability_is_mean_of_tree_probabilities():
    X, y = _data()
    rf = RandomForest(n_estimators=7, seed=2).fit(X, y)
    np.testing.assert_allclose(rf.predict_proba(X), np.mean([t.predict_proba(X) for t in rf.trees], axis=0))
    staged = rf.staged_proba(X)
    np.testing.assert_allclose(staged[-1], rf.predict_proba(X))
    np.testing.assert_allclose(staged[0], rf.trees[0].predict_proba(X))


def test_trees_are_diverse_with_bootstrap_and_feature_subsets():
    X, y = _data()
    rf = RandomForest(n_estimators=10, seed=3).fit(X, y)
    roots = {(t.feature_[0], t.threshold_[0]) for t in rf.trees}
    preds = np.array([t.predict_proba(X) for t in rf.trees])
    assert len(roots) > 1
    assert not all(np.array_equal(preds[0], p) for p in preds[1:])


def test_no_bootstrap_all_features_gives_identical_trees():
    X, y = _data()
    rf = RandomForest(n_estimators=4, bootstrap=False, max_features=None, seed=4).fit(X, y)
    single = DecisionTree().fit(X, y)
    for t in rf.trees:
        np.testing.assert_array_equal(t.feature_, single.feature_)


def test_forest_beats_single_deep_tree_on_held_out_data():
    X, y = _data(2000, seed=5)
    Xt, yt = _data(2000, seed=6)
    tree_auc = roc_auc_score(yt, DecisionTree().fit(X, y).predict_proba(Xt))
    rf_auc = roc_auc_score(yt, RandomForest(n_estimators=50, seed=5).fit(X, y).predict_proba(Xt))
    assert rf_auc > tree_auc


def test_close_to_sklearn_reference():
    """Different RNG streams mean the forests are not identical; their quality should be."""
    X, y = _data(2000, seed=7)
    Xt, yt = _data(2000, seed=8)
    ours = RandomForest(n_estimators=100, min_samples_leaf=3, seed=0).fit(X, y).predict_proba(Xt)
    sk = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=0).fit(X, y).predict_proba(Xt)[:, 1]
    assert abs(roc_auc_score(yt, ours) - roc_auc_score(yt, sk)) < 0.01
    assert np.corrcoef(ours, sk)[0, 1] > 0.97


def test_oob_estimate_is_reasonable():
    X, y = _data(1500, seed=9)
    rf = RandomForest(n_estimators=60, oob_score=True, min_samples_leaf=3, seed=9).fit(X, y)
    ok = ~np.isnan(rf.oob_proba_)
    assert ok.mean() > 0.99
    Xt, yt = _data(1500, seed=10)
    oob_auc = roc_auc_score(y[ok], rf.oob_proba_[ok])
    assert abs(oob_auc - roc_auc_score(yt, rf.predict_proba(Xt))) < 0.05
    with pytest.raises(ValueError):
        RandomForest(oob_score=True, bootstrap=False)


def test_deterministic_with_seed_and_threshold():
    X, y = _data()
    a = RandomForest(n_estimators=8, seed=11).fit(X, y).predict_proba(X)
    b = RandomForest(n_estimators=8, seed=11).fit(X, y).predict_proba(X)
    c = RandomForest(n_estimators=8, seed=12).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    rf = RandomForest(n_estimators=8, seed=11).fit(X, y)
    assert rf.predict(X, 0.2).sum() >= rf.predict(X, 0.8).sum()


def test_save_load_roundtrip(tmp_path):
    X, y = _data()
    rf = RandomForest(n_estimators=6, max_depth=5, seed=13).fit(X, y)
    rf.save(tmp_path / "rf.json", extra={"threshold": 0.45})
    rf2 = RandomForest.load(tmp_path / "rf.json")
    np.testing.assert_array_equal(rf.predict_proba(X), rf2.predict_proba(X))
    np.testing.assert_array_equal(rf.feature_importances_, rf2.feature_importances_)


def test_invalid_usage():
    X, y = _data(50)
    with pytest.raises(RuntimeError):
        RandomForest().predict_proba(X)
    with pytest.raises(ValueError):
        RandomForest(n_estimators=0)
    rf = RandomForest(n_estimators=2).fit(X, y)
    with pytest.raises(ValueError):
        rf.predict_proba(X[:, :3])
