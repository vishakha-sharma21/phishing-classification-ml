"""Tests for saved-artifact inference (the app's 8-input Logistic Regression) and Streamlit smoke tests."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DIR

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SUBMIT = "FormSubmitter:check-Check website"
pytestmark = pytest.mark.skipif(not (MODELS_DIR / "quick_logistic_regression.json").exists(), reason="models not trained")


@pytest.fixture(scope="module")
def art():
    from src.inference import load_quick_artifacts

    return load_quick_artifacts()


@pytest.fixture(scope="module")
def val_rows():
    cleaned = pd.read_csv(PROCESSED_DIR / "cleaned.csv")
    idx = json.loads((PROCESSED_DIR / "splits.json").read_text())["val"]
    return cleaned.iloc[idx[:50]]


def test_quick_model_uses_top8_training_mi_features(art):
    res = json.loads((Path(APP).parent / "reports" / "results.json").read_text(encoding="utf-8"))
    mi = res["quick_models"]["mutual_information_bits_train"]
    assert len(art.features) == 8
    assert art.features == sorted(mi, key=mi.get, reverse=True)[:8]


def test_predictions_match_model_and_zones(art, val_rows):
    from src.inference import predict_quick

    out = predict_quick(val_rows, art)
    assert list(out.columns) == ["phishing_probability", "verdict"]
    p = art.lr.predict_proba(art.lr_encoder.transform(val_rows))
    np.testing.assert_allclose(out["phishing_probability"], p.round(4))
    z = art.lr_zones
    expected = np.where(p >= z["high"], "Likely phishing", np.where(p >= z["low"], "Suspicious", "Safe"))
    assert (out["verdict"].to_numpy() == expected).all()


def test_contributions_sum_to_logit(art, val_rows):
    from src.inference import lr_contributions

    for _, row in val_rows.head(10).iterrows():
        answers = {f: int(row[f]) for f in art.features}
        c = lr_contributions(answers, art)
        p = float(art.lr.predict_proba(art.lr_encoder.transform(pd.DataFrame([answers])))[0])
        assert c.sum() + art.lr_baseline_logit == pytest.approx(math.log(p / (1 - p)), abs=1e-6)
        assert list(c.abs()) == sorted(c.abs(), reverse=True)


def test_every_answer_has_question_and_reason(art):
    from src.feature_info import QUICK_QUESTIONS

    for f in art.features:
        assert set(art.allowed_values[f]) <= set(QUICK_QUESTIONS[f]["options"]) == set(QUICK_QUESTIONS[f]["reasons"])


def test_invalid_inputs_rejected(art, val_rows):
    from src.inference import predict_quick

    good = val_rows[art.features].head(2).copy()
    bad_value = good.copy()
    bad_value.iloc[0, 0] = 7
    missing_col = good.drop(columns=art.features[3])
    missing_val = good.astype(object)
    missing_val.iloc[1, 2] = None
    for df, msg in ((bad_value, "invalid value"), (missing_col, "missing columns"),
                    (missing_val, "missing or non-numeric"), (good.iloc[0:0], "no rows")):
        with pytest.raises(ValueError, match=msg):
            predict_quick(df, art)


def test_streamlit_app_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 3
    assert len(at.selectbox) == 8
    assert len(at.sidebar) == 0  # no limitations sidebar
    # Submitting with nothing selected must be rejected with an error, not a prediction.
    at.button(key=SUBMIT).click().run()
    assert not at.exception
    assert any("Missing (8)" in e.value for e in at.error)


def test_streamlit_prediction_shows_gauge_and_reason(art, val_rows):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    row = val_rows.iloc[0]
    for sb in at.selectbox:
        sb.set_value(int(row[sb.key.removeprefix("q_")]))
    at.button(key=SUBMIT).click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert any(v in text for v in ("Safe", "Suspicious", "Likely phishing"))
    assert "**Main reason:**" in text
    assert "Logistic Regression phishing probability" in " ".join(c.value for c in at.caption)
    assert not any("agree" in m.value for m in list(at.success) + list(at.warning))  # single model: no badge
