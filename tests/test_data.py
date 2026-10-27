"""Checks that the saved raw dataset matches what Phase 1 verified."""

import pandas as pd
import pytest

from src.config import EXPECTED_FEATURE_VALUES, RAW_DATA_PATH, TARGET_COLUMN
from src.data_loading import load_raw
from src.inspect_dataset import audit_duplicates

pytestmark = pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw dataset not downloaded")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_raw()


def test_shape_and_target(df):
    assert df.shape == (11055, 31)
    assert df.columns[-1] == TARGET_COLUMN
    assert set(df[TARGET_COLUMN].unique()) == {-1, 1}


def test_values_and_missing(df):
    assert df.isna().sum().sum() == 0
    for col in df.columns:
        assert set(df[col].unique()) <= EXPECTED_FEATURE_VALUES, col


def test_duplicate_audit_is_consistent(df):
    features = [c for c in df.columns if c != TARGET_COLUMN]
    a = audit_duplicates(df, features)
    cleaned = a.pop("_cleaned")
    assert a["original_rows"] - a["exact_duplicate_rows_beyond_first"] == a["unique_rows_features_plus_label"]
    # Each conflicting vector contributes exactly two unique rows (one per label).
    assert a["unique_rows_in_conflicting_groups"] == 2 * a["conflicting_feature_vector_groups"]
    assert len(cleaned) == a["unique_rows_features_plus_label"] - a["unique_rows_in_conflicting_groups"]
    assert not cleaned.duplicated().any()
    assert not cleaned[features].duplicated().any()


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")
