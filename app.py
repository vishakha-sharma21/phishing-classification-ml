"""Educational Streamlit demo: phishing-risk estimates from the custom models.

Run with:  streamlit run app.py
Requires the artifacts written by ``python -m src.train`` (models/ and reports/).
The app never retrains, never fetches URLs, and only accepts the 30 pre-extracted
UCI features.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, PROCESSED_DIR
from src.feature_info import FEATURES
from src.inference import Artifacts, load_artifacts, predict, prepare_input

st.set_page_config(page_title="Phishing risk (UCI features)", page_icon="🛡️", layout="wide")

LIMITATIONS = """
* **Model-estimated risk, not a guarantee.** A low probability does not mean a site is safe.
* **Pre-extracted features only.** The models expect the 30 engineered indicators of the UCI dataset
  (collected c. 2012). This app does **not** analyse live URLs, and it never visits any website.
* **Old data.** Several inputs depend on services that no longer exist or have changed (Alexa traffic rank,
  public PageRank) and phishing techniques have evolved; performance on today's web is unknown.
* **Population.** Models were trained on 4,004 *unique feature vectors* (duplicates removed), not on a sample of
  real web traffic, so probabilities are not calibrated to how common phishing is in practice.
* **Coding ambiguities.** The meaning of some codes is not documented by the dataset authors
  (e.g. `redirect`, `prefix_suffix`, `domain_registration_length`; see reports/features.md).
"""


@st.cache_resource
def get_artifacts() -> Artifacts:
    return load_artifacts()


def figure(name: str, caption: str) -> None:
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure {name} not found — run `python -m src.train`.")


def show_predictions(out: pd.DataFrame, art: Artifacts) -> None:
    c1, c2 = st.columns(2)
    for col, key, label, thr in ((c1, "lr", "Logistic Regression (custom)", art.lr_threshold),
                                 (c2, "rf", "Random Forest (custom)", art.rf_threshold)):
        p = float(out[f"{key}_phishing_probability"].iloc[0])
        pred = out[f"{key}_prediction"].iloc[0]
        with col:
            st.subheader(label)
            st.metric("Estimated phishing probability", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
            st.write(f"Operating threshold: **{thr:.2f}** (chosen to maximise validation F1)")
            if pred == "phishing":
                st.error("⚠️ Predicted class: **phishing**")
            else:
                st.success("✅ Predicted class: **legitimate**")
    if out["lr_prediction"].iloc[0] != out["rf_prediction"].iloc[0]:
        st.warning("The two models disagree for this input — treat the estimate as uncertain.")


def manual_tab(art: Artifacts) -> None:
    st.markdown("Choose a value for **every** feature. Codes follow the dataset convention "
                "(−1 = phishing-like, 0 = suspicious, 1 = legitimate-like); see each option's meaning.")
    example = None
    split_file = PROCESSED_DIR / "cleaned.csv"
    if split_file.exists() and st.button("Load a random example from the validation split"):
        cleaned = pd.read_csv(split_file)
        val_idx = json.loads((PROCESSED_DIR / "splits.json").read_text())["val"]
        row = cleaned.iloc[val_idx].sample(1, random_state=None).iloc[0]
        example = {f: int(row[f]) for f in art.feature_order}
        st.session_state["example_label"] = "phishing" if row["phishing"] == 1 else "legitimate"
        for f, v in example.items():
            st.session_state[f"feat_{f}"] = v

    values: dict[str, int | None] = {}
    groups: dict[str, list[str]] = {}
    for f in art.feature_order:
        groups.setdefault(FEATURES[f]["group"], []).append(f)
    with st.form("manual"):
        for group, feats in groups.items():
            st.markdown(f"#### {group}")
            cols = st.columns(3)
            for i, f in enumerate(feats):
                info = FEATURES[f]
                opts = art.allowed_values[f]
                with cols[i % 3]:
                    values[f] = st.selectbox(
                        info["label"], opts, index=None, key=f"feat_{f}", placeholder="Select…",
                        format_func=lambda v, info=info: f"{v}: {info['values'].get(v, 'code ' + str(v))}",
                        help=f"`{f}` — {info['rule']}" + (f"\n\n⚠️ {info['note']}" if info.get("note") else ""))
        submitted = st.form_submit_button("Estimate phishing risk", type="primary")
    if submitted:
        missing = [FEATURES[f]["label"] for f, v in values.items() if v is None]
        if missing:
            st.error(f"Please choose a value for all features. Missing ({len(missing)}): " + ", ".join(missing))
            return
        out = predict(pd.DataFrame([values]), art)
        show_predictions(out, art)
        if "example_label" in st.session_state:
            st.caption(f"True label of the loaded validation example: {st.session_state.pop('example_label')}")


def batch_tab(art: Artifacts) -> None:
    st.markdown(f"Upload a CSV with exactly these {len(art.feature_order)} columns (any order; extra columns are "
                "ignored and kept in the output). Every value must be one of the allowed integer codes.")
    template = pd.DataFrame(columns=art.feature_order).to_csv(index=False)
    st.download_button("Download empty CSV template", template, "phishing_features_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is None:
        return
    try:
        df = pd.read_csv(up)
    except Exception as exc:  # malformed file
        st.error(f"Could not read the CSV: {exc}")
        return
    _, problems = prepare_input(df, art)
    if problems:
        st.error("The file was rejected:\n\n" + "\n".join(f"* {p}" for p in problems[:20]))
        return
    out = pd.concat([df, predict(df, art)], axis=1)
    st.success(f"Scored {len(out)} row(s).")
    st.dataframe(out, use_container_width=True)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button("Download predictions", buf.getvalue(), "predictions.csv", "text/csv")


def performance_tab(art: Artifacts) -> None:
    ev = art.results.get("evaluation")
    if not ev:
        st.info("No evaluation results found — run `python -m src.train`.")
        return
    st.markdown(f"Held-out **test set** ({ev['test_rows']} unique rows), evaluated once after model selection on the "
                "validation set. Positive class = phishing. 95% bootstrap intervals in brackets.")
    rows = []
    for name in ("Custom LR", "Custom RF", "Custom tree", "sklearn LR (ref)", "sklearn RF (ref)",
                 "Best single feature", "Majority"):
        m, ci = ev["test"][name], ev["test_bootstrap_ci"].get(name, {})
        fmt = lambda k: f"{m[k]:.3f}" + (f" [{ci[k]['low']:.3f}, {ci[k]['high']:.3f}]" if k in ci else "")  # noqa: E731
        rows.append({"model": name, "threshold": m["threshold"], "accuracy": fmt("accuracy"),
                     "precision": fmt("precision"), "recall": fmt("recall"), "F1": fmt("f1"),
                     "ROC-AUC": fmt("roc_auc"), "PR-AUC": fmt("pr_auc")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("“ref” rows are scikit-learn reference models, not the custom implementations.")
    c1, c2 = st.columns(2)
    with c1:
        figure("eval_02_confusion_test.png", "Confusion matrices (test)")
        figure("int_03_rf_impurity.png", "Random Forest impurity importance")
    with c2:
        figure("eval_01_roc_pr_test.png", "ROC and precision–recall curves (test)")
        figure("int_01_lr_coefficients.png", "Logistic Regression coefficients")
    figure("int_04_permutation_importance.png", "Permutation importance on validation data")


def main() -> None:
    st.title("🛡️ Phishing website risk — from-scratch ML models")
    st.caption("Logistic Regression and Random Forest implemented in NumPy, trained on the UCI Phishing Websites "
               "dataset (Mohammad, Thabtah & McCluskey, CC BY 4.0).")
    try:
        art = get_artifacts()
    except FileNotFoundError:
        st.error("Saved models not found. Run `python -m src.train` first.")
        st.stop()
    with st.sidebar:
        st.header("Limitations")
        st.markdown(LIMITATIONS)
        st.header("About URLs")
        st.markdown("This version **does not accept URLs**. Turning a live URL into these 30 indicators needs "
                    "WHOIS, DNS, page-content and reputation look-ups that are not implemented or validated here, "
                    "and visiting suspicious sites is unsafe. Enter pre-extracted features instead.")
    t1, t2, t3 = st.tabs(["Single website", "Batch CSV", "Model performance"])
    with t1:
        manual_tab(art)
    with t2:
        batch_tab(art)
    with t3:
        performance_tab(art)


if __name__ == "__main__" or Path(__file__).name == "app.py":
    main()
