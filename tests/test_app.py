"""Tests for saved-artifact inference and a Streamlit smoke test."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DIR

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SUBMIT = "FormSubmitter:manual-Estimate phishing risk"
pytestmark = pytest.mark.skipif(not (MODELS_DIR / "random_forest.json").exists(), reason="models not trained")


@pytest.fixture(scope="module")
def art():
    from src.inference import load_artifacts

    return load_artifacts()


@pytest.fixture(scope="module")
def val_rows(art):
    import json

    cleaned = pd.read_csv(PROCESSED_DIR / "cleaned.csv")
    idx = json.loads((PROCESSED_DIR / "splits.json").read_text())["val"]
    return cleaned.iloc[idx[:50]]


def test_predictions_match_models_and_thresholds(art, val_rows):
    from src.inference import predict

    out = predict(val_rows[art.feature_order], art)
    assert list(out.columns) == ["lr_phishing_probability", "lr_prediction", "rf_phishing_probability", "rf_prediction"]
    p = art.rf.predict_proba(art.rf_encoder.transform(val_rows))
    np.testing.assert_allclose(out["rf_phishing_probability"], p.round(4))
    assert ((out["rf_phishing_probability"] >= art.rf_threshold) == (out["rf_prediction"] == "phishing")).all()


def test_column_order_does_not_matter(art, val_rows):
    from src.inference import predict

    a = predict(val_rows[art.feature_order], art)
    b = predict(val_rows[art.feature_order[::-1]], art)
    pd.testing.assert_frame_equal(a, b)


def test_invalid_inputs_rejected(art, val_rows):
    from src.inference import prepare_input

    good = val_rows[art.feature_order].head(2).copy()
    bad_value = good.copy()
    bad_value.iloc[0, 0] = 7
    missing_col = good.drop(columns=art.feature_order[3])
    missing_val = good.astype(object)
    missing_val.iloc[1, 2] = None
    text = good.astype(object)
    text.iloc[0, 1] = "abc"
    for df, msg in ((bad_value, "invalid value"), (missing_col, "missing columns"),
                    (missing_val, "missing or non-numeric"), (text, "missing or non-numeric"),
                    (good.iloc[0:0], "no rows")):
        clean, problems = prepare_input(df, art)
        assert clean is None and any(msg in p for p in problems), (msg, problems)


def test_streamlit_app_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 3
    assert len(at.selectbox) == 30
    # Submitting with nothing selected must be rejected with an error, not a prediction.
    at.button(key=SUBMIT).click().run()
    assert not at.exception
    assert any("Missing (30)" in e.value for e in at.error)
    assert len(at.metric) == 0


def test_streamlit_manual_prediction(art, val_rows):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    row = val_rows.iloc[0]
    for sb in at.selectbox:
        sb.set_value(int(row[sb.key.removeprefix("feat_")]))
    at.button(key=SUBMIT).click().run()
    assert not at.exception
    assert len(at.metric) == 2
    assert "%" in at.metric[0].value
