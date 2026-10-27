"""Tests for cleaning, label mapping, the stratified split and EDA statistics."""

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_DATA_PATH
from src.eda import cramers_v, single_feature_scores
from src.preprocessing import TARGET, clean, map_labels, split_indices


def _toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 1, 1, -1, -1, 0, 0],
            "b": [1, 1, 1, -1, -1, 1, 1],
            "result": [1, 1, -1, -1, -1, 1, 1],
        }
    )


def test_clean_collapses_duplicates_and_drops_conflicts():
    out = clean(_toy())
    # (1,1) appears with both labels -> dropped; (-1,-1) and (0,1) are collapsed.
    assert len(out) == 2
    assert set(map(tuple, out[["a", "b"]].to_numpy())) == {(-1, -1), (0, 1)}
    assert out.set_index("a").loc[-1, TARGET] == 1  # original -1 -> phishing
    assert out.set_index("a").loc[0, TARGET] == 0
    assert "result" not in out.columns


def test_map_labels_rejects_unknown():
    with pytest.raises(ValueError):
        map_labels(pd.DataFrame({"a": [1], "result": [2]}))


def test_split_is_disjoint_stratified_and_deterministic():
    y = np.array([1] * 520 + [0] * 480)
    s1, s2 = split_indices(y), split_indices(y)
    assert all(np.array_equal(getattr(s1, p), getattr(s2, p)) for p in ("train", "val", "test"))
    parts = [s1.train, s1.val, s1.test]
    assert sum(map(len, parts)) == len(y)
    assert len(np.unique(np.concatenate(parts))) == len(y)
    assert [len(p) for p in parts] == [700, 150, 150]
    for p in parts:
        assert abs(y[p].mean() - 0.52) < 0.01


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw dataset not downloaded")
def test_real_split_sizes():
    from src.data_loading import load_raw

    c = clean(load_raw())
    s = split_indices(c[TARGET])
    assert len(c) == 5721
    assert (len(s.train), len(s.val), len(s.test)) == (4004, 858, 859)


def test_cramers_v_bounds():
    x = np.array([0, 1] * 200)
    assert cramers_v(x, x) == pytest.approx(1.0, abs=0.01)
    rng = np.random.default_rng(0)
    assert cramers_v(rng.integers(0, 2, 2000), rng.integers(0, 2, 2000)) < 0.05


def test_single_feature_scores_use_train_only_and_handle_unseen():
    train = pd.DataFrame({"f": [1, 1, -1, -1], TARGET: [1, 1, 0, 0]})
    other = pd.DataFrame({"f": [1, -1, 0]})
    s = single_feature_scores(train, other, "f", alpha=0.0)
    assert s[0] == 1.0 and s[1] == 0.0
    assert s[2] == pytest.approx(0.5)  # unseen value -> training prior
