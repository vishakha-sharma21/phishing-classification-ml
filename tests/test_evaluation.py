"""Tests for metrics, bootstrap utilities and threshold tuning."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import bootstrap_ci, compute_metrics, paired_bootstrap_diff, verdict
from src.threshold_tuning import assign_zone, threshold_curve, tune_threshold, zone_table, zone_thresholds


def test_compute_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["accuracy"] == 0.5
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))  # uses probabilities, not labels
    assert compute_metrics(y, p, 0.35)["recall"] == 1.0


def test_compute_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0.5]))


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    ci = bootstrap_ci(y, p, 0.5, n_boot=200)
    for v in ci.values():
        assert v["low"] <= v["point"] <= v["high"]


def test_paired_diff_of_identical_models_is_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    d = paired_bootstrap_diff(y, p, 0.5, p, 0.5, n_boot=100)
    assert all(v["low"] == v["high"] == 0 for v in d.values())
    assert verdict(d["f1"]) == "no meaningful difference"
    assert verdict({"low": 0.02, "high": 0.05, "mean": 0.03}) == "clear improvement"
    assert verdict({"low": -0.05, "high": -0.01, "mean": -0.03}) == "worse performance"


def test_threshold_tuning():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.35, 0.8, 0.9])
    t = tune_threshold(y, p, "f1")
    assert 0.3 < t <= 0.35
    curve = threshold_curve(y, p)
    assert len(curve["threshold"]) == len(curve["f1"]) == 91


def test_zone_thresholds_meet_their_targets():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    z = zone_thresholds(y, p, target=0.95)
    assert z["low"] <= z["high"]
    assert compute_metrics(y, p, z["low"])["recall"] >= 0.95          # few phishing sites called Safe
    assert compute_metrics(y, p, z["high"])["precision"] >= 0.95      # few false "Likely phishing"
    assert compute_metrics(y, p, z["low"] + 0.01)["recall"] < 0.95    # low is the largest such cut-off


def test_assign_zone_boundaries_and_table():
    p = np.array([0.0, 0.199, 0.2, 0.5, 0.9, 1.0])
    assert assign_zone(p, 0.2, 0.9).tolist() == [0, 0, 1, 1, 2, 2]
    t = zone_table(np.array([0, 0, 1, 0, 1, 1]), p, 0.2, 0.9)
    assert [t[k]["rows"] for k in ("Safe", "Suspicious", "Likely phishing")] == [2, 2, 2]
    assert t["Likely phishing"]["phishing_rate"] == 1.0 and t["Safe"]["phishing_rate"] == 0.0
