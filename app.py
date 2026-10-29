"""Streamlit demo: phishing-risk estimates from the custom 8-input Logistic Regression.

Run with:  streamlit run app.py
Requires the artifacts written by ``python -m src.train`` (models/ and reports/).
The app never retrains, never fetches URLs, and only accepts the 8 pre-extracted UCI
features chosen by training-set mutual information.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, PROCESSED_DIR
from src.feature_info import FEATURES, QUICK_QUESTIONS
from src.inference import QuickArtifacts, lr_contributions, load_quick_artifacts, predict_quick

st.set_page_config(page_title="Phishing risk checker", page_icon="🛡️", layout="wide")

GREEN, AMBER, RED = "#2e7d32", "#f9a825", "#c62828"
VERDICT_STYLE = {"Safe": (GREEN, "✅"), "Suspicious": (AMBER, "⚠️"), "Likely phishing": (RED, "⛔")}


@st.cache_resource
def get_artifacts() -> QuickArtifacts:
    return load_quick_artifacts()


def figure(name: str, caption: str) -> None:
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


# ------------------------------------------------------------------ result widgets


def risk_gauge(p: float, zones: dict[str, float], verdict: str) -> None:
    """Colour meter: green / amber / red bands at the validation thresholds, marker at p."""
    low, high = zones["low"] * 100, zones["high"] * 100
    colour, icon = VERDICT_STYLE[verdict]
    pos = min(max(p * 100, 0.5), 99.5)
    st.markdown(f"""
<div style="font-size:3rem;font-weight:700;color:{colour};line-height:1.1">{p:.1%}</div>
<div style="font-size:1.3rem;font-weight:600;color:{colour};margin-bottom:.6rem">{icon} {verdict}</div>
<div style="position:relative;height:22px;border-radius:11px;overflow:hidden;display:flex">
  <div style="width:{low}%;background:{GREEN}"></div>
  <div style="width:{high - low}%;background:{AMBER}"></div>
  <div style="width:{100 - high}%;background:{RED}"></div>
</div>
<div style="position:relative;height:16px">
  <div style="position:absolute;left:{pos}%;top:-30px;transform:translateX(-50%);width:4px;height:30px;
       background:#111;border:1px solid #fff;border-radius:2px"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:.8rem;color:#666">
  <span>0% · Safe</span><span>Suspicious from {low:.0f}%</span><span>Likely phishing from {high:.0f}%</span>
  <span>100%</span>
</div>""", unsafe_allow_html=True)
    st.caption("Logistic Regression phishing probability (8 inputs)")


def why_panel(answers: dict[str, int], art: QuickArtifacts, p_lr: float) -> None:
    """Bar chart of each answer's push on the Logistic Regression score, plus the main reason."""
    contrib = lr_contributions(answers, art)
    towards_phishing = p_lr >= art.lr_threshold
    side = contrib[contrib > 0] if towards_phishing else contrib[contrib < 0]
    top = (side if len(side) else contrib).abs().idxmax()
    st.markdown(f"**Main reason:** {QUICK_QUESTIONS[top]['reasons'][answers[top]]}.")

    df = pd.DataFrame({
        "answer": [QUICK_QUESTIONS[f]["options"][answers[f]] for f in contrib.index],
        "question": [QUICK_QUESTIONS[f]["question"] for f in contrib.index],
        "contribution": contrib.values.round(3),
        "direction": np.where(contrib.values > 0, "towards phishing", "towards legitimate"),
    })
    df["label"] = [f"{FEATURES[f]['label']}: {a}" for f, a in zip(contrib.index, df["answer"])]
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("contribution:Q", title="Push on the Logistic Regression score (log-odds)"),
        y=alt.Y("label:N", sort=list(df["label"]), title=None, axis=alt.Axis(labelLimit=420)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["towards phishing", "towards legitimate"],
                                                        range=[RED, GREEN]), legend=alt.Legend(title=None,
                                                                                               orient="bottom")),
        tooltip=["question", "answer", "contribution", "direction"],
    ).properties(height=38 * len(df))
    st.altair_chart(chart, width="stretch")
    st.caption("Each bar is weight × (your answer − the average training site) in the Logistic Regression model. "
               "Red pushes towards phishing, green towards legitimate; bars are sorted by size.")


# ------------------------------------------------------------------ tabs


def check_tab(art: QuickArtifacts) -> None:
    st.markdown("Answer the 8 questions about the website, then press **Check website**.")
    cleaned_path = PROCESSED_DIR / "cleaned.csv"
    if cleaned_path.exists():
        c1, c2, _ = st.columns([1, 1, 2])
        for col, label, cls in ((c1, "Load example phishing site", 1), (c2, "Load example legitimate site", 0)):
            if col.button(label, key=f"example_{cls}"):
                cleaned = pd.read_csv(cleaned_path)
                val = cleaned.iloc[json.loads((PROCESSED_DIR / "splits.json").read_text())["val"]]
                row = val[val["phishing"] == cls].sample(1).iloc[0]
                for f in art.features:
                    st.session_state[f"q_{f}"] = int(row[f])
                st.session_state["example_label"] = "phishing" if cls else "legitimate"

    answers: dict[str, int | None] = {}
    with st.form("check"):
        cols = st.columns(2)
        for i, f in enumerate(art.features):
            q = QUICK_QUESTIONS[f]
            opts = [v for v in q["options"] if v in art.allowed_values[f]]
            with cols[i % 2]:
                answers[f] = st.selectbox(q["question"], opts, index=None, key=f"q_{f}", placeholder="Choose an answer…",
                                          format_func=lambda v, q=q: q["options"][v])
        submitted = st.form_submit_button("Check website", type="primary")
    if not submitted:
        return
    missing = [QUICK_QUESTIONS[f]["question"] for f, v in answers.items() if v is None]
    if missing:
        st.error(f"Please answer every question. Missing ({len(missing)}): " + " · ".join(missing))
        return

    out = predict_quick(pd.DataFrame([answers]), art).iloc[0]
    p_lr = float(out["phishing_probability"])
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Risk")
        risk_gauge(p_lr, art.lr_zones, out["verdict"])
        if "example_label" in st.session_state:
            st.caption(f"Loaded example's true label: {st.session_state.pop('example_label')}")
    with right:
        st.subheader("Why this result?")
        why_panel({f: int(v) for f, v in answers.items()}, art, p_lr)
    with st.expander("How the verdict is decided"):
        st.markdown(
            f"- **Safe** below {art.lr_zones['low']:.0%}, **Suspicious** from {art.lr_zones['low']:.0%}, "
            f"**Likely phishing** from {art.lr_zones['high']:.0%} (Logistic Regression probability).\n"
            f"- Both cut-offs were chosen on the validation set only: the lower one keeps at least "
            f"{art.lr_zones['target']:.0%} of phishing sites out of *Safe*; the upper one makes at least "
            f"{art.lr_zones['target']:.0%} of *Likely phishing* sites real phishing.")


def batch_tab(art: QuickArtifacts) -> None:
    st.markdown(f"Upload a CSV with these {len(art.features)} columns (any order; extra columns are kept in the "
                "output). Each value must be one of the dataset codes −1, 0 or 1 allowed for that column.")
    st.code(", ".join(art.features), language=None)
    template = pd.DataFrame(columns=art.features).to_csv(index=False)
    st.download_button("Download empty CSV template", template, "phishing_quick_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is None:
        return
    try:
        df = pd.read_csv(up)
    except Exception as exc:  # malformed file
        st.error(f"Could not read the CSV: {exc}")
        return
    try:
        scored = predict_quick(df, art)
    except ValueError as exc:
        st.error("The file was rejected:\n\n" + "\n".join(f"* {p}" for p in str(exc).split("; ")[:20]))
        return
    out = pd.concat([df, scored], axis=1)
    st.success(f"Scored {len(out)} row(s): " + ", ".join(f"{v} {k}" for k, v in out["verdict"].value_counts().items()))
    st.dataframe(out, width="stretch")
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button("Download predictions", buf.getvalue(), "predictions.csv", "text/csv")


def insights_tab(art: QuickArtifacts) -> None:
    q, ev = art.results.get("quick_models"), art.results.get("evaluation")
    if not q or not ev:
        st.info("No results found — run `python -m src.train`.")
        return
    st.markdown("### 8 inputs vs all 30\nHeld-out **test set** (859 sites), evaluated once after all choices were made "
                "on the validation set. 95% bootstrap intervals in brackets; positive class = phishing.")
    rows = []
    for name, m, ci in (("Logistic Regression — 8 inputs (app)", q["test"]["Quick LR (8)"], q["test_bootstrap_ci"]["Quick LR (8)"]),
                        ("Logistic Regression — 30 inputs", q["test"]["Full LR (30)"], q["test_bootstrap_ci"]["Full LR (30)"]),
                        ("Best single feature (baseline)", ev["test"]["Best single feature"],
                         ev["test_bootstrap_ci"]["Best single feature"]),
                        ("Majority class (baseline)", ev["test"]["Majority"], {})):
        fmt = lambda k: f"{m[k]:.3f}" + (f" [{ci[k]['low']:.3f}, {ci[k]['high']:.3f}]" if k in ci else "")  # noqa: E731
        rows.append({"model": name, "threshold": f"{m['threshold']:.2f}", "accuracy": fmt("accuracy"),
                     "precision": fmt("precision"), "recall": fmt("recall"), "F1": fmt("f1"), "ROC-AUC": fmt("roc_auc")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    d = q["paired_comparisons_test"]["Quick LR (8) vs Full LR (30)"]
    st.markdown(f"- **8 vs 30 inputs:** F1 {d['f1']['mean']:+.3f} [{d['f1']['low']:+.3f}, {d['f1']['high']:+.3f}] — "
                f"*{d['f1']['verdict']}*; ROC-AUC {d['roc_auc']['mean']:+.3f} [{d['roc_auc']['low']:+.3f}, "
                f"{d['roc_auc']['high']:+.3f}] — *{d['roc_auc']['verdict']}*.")
    b = ev["paired_comparisons_test"]["Custom LR vs Best single feature"]["f1"]
    st.markdown(f"- **30-input model vs best single feature:** F1 {b['mean']:+.3f} [{b['low']:+.3f}, {b['high']:+.3f}] "
                f"— *{b['verdict']}*.")

    st.markdown("### Verdict zones on the test set")
    z = q["zones"]
    st.markdown(f"Cut-offs (chosen on validation): **{z['thresholds']['lr']['low']:.2f}** and "
                f"**{z['thresholds']['lr']['high']:.2f}**.")
    zt = pd.DataFrame([{"zone": zone, "sites": r["rows"], "share": f"{r['share']:.1%}",
                        "phishing rate": f"{r['phishing_rate']:.1%}"} for zone, r in z["test"]["Quick LR (8)"].items()])
    st.dataframe(zt, hide_index=True, width="stretch")

    st.markdown("### The 8 inputs")
    mi = q["mutual_information_bits_train"]
    st.dataframe(pd.DataFrame({"input": [QUICK_QUESTIONS[f]["question"] for f in q["features"]],
                               "feature": q["features"],
                               "mutual information (bits, training data)": [mi[f] for f in q["features"]]}),
                 hide_index=True, width="stretch")

    st.markdown("### 30-input Logistic Regression")
    c1, c2 = st.columns(2)
    with c1:
        figure("lr_06_confusion_test.png", "Confusion matrices (test)")
        figure("int_01_lr_coefficients.png", "Logistic Regression coefficients")
    with c2:
        figure("lr_05_roc_pr_test.png", "ROC and precision–recall curves (test)")
        figure("lr_09_mi_top_k.png", "Validation ROC-AUC against the number of inputs")


def main() -> None:
    st.title("🛡️ Phishing website risk checker")
    st.caption("Logistic Regression implemented from scratch in NumPy, trained on the UCI Phishing "
               "Websites dataset (Mohammad, Thabtah & McCluskey, CC BY 4.0).")
    try:
        art = get_artifacts()
    except FileNotFoundError:
        st.error("Saved models not found. Run `python -m src.train` first.")
        st.stop()
    t1, t2, t3 = st.tabs(["Check a website", "Batch CSV", "Model insights"])
    with t1:
        check_tab(art)
    with t2:
        batch_tab(art)
    with t3:
        insights_tab(art)


if __name__ == "__main__" or Path(__file__).name == "app.py":
    main()
