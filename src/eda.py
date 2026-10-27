"""Phase 2 — exploratory data analysis: figures and summary statistics.

Rules used throughout:
* Label-free summaries (value counts, dimensions) use the full cleaned data.
* Label-dependent summaries (phishing rates, associations) use the TRAIN partition only.
* Single-feature predictive diagnostics are fitted on TRAIN and scored on VALIDATION.
* The TEST partition is never read here beyond its size.

Run with ``python -m src.eda``; writes figures to reports/figures/ and
reports/eda_results.json.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score, roc_auc_score

from src.config import FIGURES_DIR, RANDOM_SEED, REPORTS_DIR, TARGET_COLUMN
from src.data_loading import load_raw
from src.preprocessing import TARGET, clean, feature_columns, map_labels, split_indices

# Palette (validated reference slots): identity colours for the two classes.
PHISH = "#eb6834"
LEGIT = "#2a78d6"
VALUE_COLORS = {-1: "#eb6834", 0: "#eda100", 1: "#2a78d6"}
VALUE_MARKERS = {-1: "o", 0: "s", 1: "D"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9,
            "legend.frameon": False,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPORTS_DIR.parent)).replace("\\", "/")


# --------------------------------------------------------------------------- data


def load_eda_data() -> dict[str, Any]:
    """Raw frame, cleaned frame and the approved split of the cleaned frame."""
    raw = load_raw()
    cleaned = clean(raw)
    split = split_indices(cleaned[TARGET])
    return {
        "raw": raw,
        "cleaned": cleaned,
        "features": feature_columns(cleaned),
        "split": split,
        "train": cleaned.iloc[split.train],
        "val": cleaned.iloc[split.val],
    }


# ---------------------------------------------------------------------- statistics


def cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013) between two discrete variables."""
    table = pd.crosstab(x, y).to_numpy()
    if min(table.shape) < 2:
        return 0.0
    n = table.sum()
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    r, k = table.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def association_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pairwise bias-corrected Cramér's V for the given discrete columns."""
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            m.loc[a, b] = m.loc[b, a] = cramers_v(df[a].to_numpy(), df[b].to_numpy())
    return m


def target_association(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-feature association with the label on the training partition."""
    y = train[TARGET].to_numpy()
    rows = []
    for f in features:
        x = train[f].to_numpy()
        rates = train.groupby(f)[TARGET].agg(["mean", "size"])
        rows.append(
            {
                "feature": f,
                "cramers_v": round(cramers_v(x, y), 4),
                "mutual_info_bits": round(mutual_info_score(x, y) / np.log(2), 4),
                "phishing_rate_by_value": {int(k): round(float(v), 4) for k, v in rates["mean"].items()},
                "n_by_value": {int(k): int(v) for k, v in rates["size"].items()},
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def single_feature_scores(
    train: pd.DataFrame, other: pd.DataFrame, feature: str, alpha: float = 1.0
) -> np.ndarray:
    """Score rows of ``other`` by the Laplace-smoothed TRAIN phishing rate of their value.

    Unseen values fall back to the training prior.
    """
    g = train.groupby(feature)[TARGET].agg(["sum", "size"])
    prior = train[TARGET].mean()
    rate = (g["sum"] + alpha * prior) / (g["size"] + alpha)
    return other[feature].map(rate).fillna(prior).to_numpy()


def single_feature_auc(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Validation ROC-AUC of each one-feature rate model, with a 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    y = val[TARGET].to_numpy()
    boot = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rows = []
    for f in features:
        s = single_feature_scores(train, val, f)
        auc = roc_auc_score(y, s)
        bs = [roc_auc_score(y[b], s[b]) for b in boot if 0 < y[b].sum() < len(b)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        train_auc = roc_auc_score(train[TARGET], single_feature_scores(train, train, f))
        rows.append(
            {
                "feature": f,
                "val_roc_auc": round(float(auc), 4),
                "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
                "train_roc_auc": round(float(train_auc), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)


def duplicate_leakage_diagnostic(raw: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """What happens if the SAME split procedure is applied to the raw, un-deduplicated data.

    Measures how many validation rows have an exact feature-vector twin in train, and
    how accurate a pure look-up ("memorise the training label") is on those rows.
    Diagnostic only; the raw split is never used for model selection.
    """
    df = map_labels(raw)
    split = split_indices(df[TARGET])
    tr, va = df.iloc[split.train], df.iloc[split.val]
    key = lambda d: d[features].astype(str).agg("".join, axis=1)  # noqa: E731
    tr_key, va_key = key(tr), key(va)
    lookup = tr.assign(_k=tr_key.values).groupby("_k")[TARGET].mean()
    seen = va_key.isin(lookup.index).to_numpy()
    pred = (va_key[seen].map(lookup).to_numpy() >= 0.5).astype(int)
    acc_seen = float((pred == va[TARGET].to_numpy()[seen]).mean())
    return {
        "raw_val_rows": int(len(va)),
        "raw_val_rows_with_exact_twin_in_train": int(seen.sum()),
        "raw_val_overlap_percent": round(100 * seen.mean(), 2),
        "lookup_accuracy_on_overlapping_val_rows": round(acc_seen, 4),
        "cleaned_val_overlap_percent": 0.0,
    }


# ------------------------------------------------------------------------- figures


def fig_class_distribution(raw: pd.DataFrame, cleaned: pd.DataFrame) -> str:
    """Grouped bars: phishing/legitimate counts before and after cleaning."""
    raw_y = map_labels(raw)[TARGET]
    data = {"Raw (11,055 rows)": raw_y, "Cleaned": cleaned[TARGET]}
    data = {f"Raw\n(n={len(raw_y):,})": raw_y, f"Cleaned\n(n={len(cleaned):,})": cleaned[TARGET]}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(data))
    w = 0.36
    for off, cls, color, name in [(-w / 2, 1, PHISH, "Phishing"), (w / 2, 0, LEGIT, "Legitimate")]:
        counts = [int((y == cls).sum()) for y in data.values()]
        pcts = [100 * (y == cls).mean() for y in data.values()]
        bars = ax.bar(x + off, counts, w - 0.03, color=color, label=name)
        for b, c, p in zip(bars, counts, pcts):
            ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n{p:.1f}%", ha="center", va="bottom",
                    color=INK, fontsize=8)
    ax.set_xticks(x, list(data))
    ax.set_ylabel("Rows")
    ax.set_ylim(0, max(len(raw_y) * 0.68, 1))
    ax.grid(axis="x", visible=False)
    ax.set_title("Class distribution before and after duplicate cleaning")
    ax.legend(loc="upper right")
    return _save(fig, "01_class_distribution.png")


def fig_duplicates(raw: pd.DataFrame, audit: dict[str, Any]) -> str:
    """Left: row counts through the cleaning funnel. Right: how often each vector repeats."""
    features = [c for c in raw.columns if c != TARGET_COLUMN]
    sizes = raw.groupby(features).size()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    labels = ["Original rows", "Unique rows\n(features+label)", "Unique feature\nvectors", "Retained\n(policy)"]
    vals = [audit["original_rows"], audit["unique_rows_features_plus_label"],
            audit["unique_feature_vectors"], audit["retained_rows_after_policy"]]
    bars = a1.barh(labels[::-1], vals[::-1], color=LEGIT, height=0.6)
    for b, v in zip(bars, vals[::-1]):
        a1.text(v, b.get_y() + b.get_height() / 2, f" {v:,}", va="center", color=INK, fontsize=8)
    a1.set_xlim(0, max(vals) * 1.15)
    a1.grid(axis="y", visible=False)
    a1.set_title("Cleaning funnel")
    vc = sizes.value_counts().sort_index()
    a2.bar(vc.index, vc.values, color=LEGIT, width=0.8)
    a2.set_yscale("log")
    a2.set_xlabel("Times a feature vector occurs in the raw data")
    a2.set_ylabel("Number of distinct vectors (log)")
    a2.grid(axis="x", visible=False)
    a2.set_title("Repetition of identical feature vectors")
    fig.tight_layout()
    return _save(fig, "02_duplicate_statistics.png")


def fig_stacked_by_class(train: pd.DataFrame, features: list[str]) -> str:
    """Small multiples: count of each feature value, stacked by class (TRAIN)."""
    fig, axes = plt.subplots(6, 5, figsize=(14, 15))
    for ax, f in zip(axes.flat, features):
        tab = pd.crosstab(train[f], train[TARGET]).reindex(columns=[1, 0], fill_value=0)
        xs = [str(v) for v in tab.index]
        ax.bar(xs, tab[1], color=PHISH, width=0.6, edgecolor=SURFACE, linewidth=1)
        ax.bar(xs, tab[0], bottom=tab[1], color=LEGIT, width=0.6, edgecolor=SURFACE, linewidth=1)
        ax.set_title(f, fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (PHISH, LEGIT)]
    fig.legend(handles, ["Phishing", "Legitimate"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Feature values by class — training partition (n={len(train):,})", y=1.015,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "03_feature_values_by_class.png")


def fig_phishing_rate(assoc: pd.DataFrame, prior: float) -> str:
    """Dot plot: phishing rate for each value of each feature (TRAIN), sorted by Cramér's V."""
    a = assoc.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    for i, r in a.iterrows():
        rates = r["phishing_rate_by_value"]
        ax.plot([min(rates.values()), max(rates.values())], [i, i], color=GRID, lw=2, zorder=1)
        for v, rate in rates.items():
            n = r["n_by_value"][v]
            ax.scatter(rate, i, s=20 + 120 * n / 4000, color=VALUE_COLORS[v], marker=VALUE_MARKERS[v],
                       edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.axvline(prior, color=INK_2, ls="--", lw=1)
    ax.text(prior, len(a) - 0.3, f" overall {prior:.2f}", color=INK_2, fontsize=8, va="bottom")
    ax.set_yticks(range(len(a)), a["feature"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.7, len(a) + 0.3)
    ax.set_xlabel("Phishing rate among training rows with this value")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], ls="", marker=VALUE_MARKERS[v], color=VALUE_COLORS[v], markersize=7)
               for v in (-1, 0, 1)]
    ax.legend(handles, ["value -1", "value 0", "value 1"], loc="lower right", title="marker size ∝ rows")
    ax.set_title("Phishing rate per feature value (train; sorted by Cramér's V)")
    return _save(fig, "04_phishing_rate_by_value.png")


def fig_association_heatmap(assoc_m: pd.DataFrame) -> str:
    """Heatmap of bias-corrected Cramér's V among features and the label (TRAIN)."""
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(assoc_m.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(assoc_m)), assoc_m.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(assoc_m)), assoc_m.index, fontsize=7)
    ax.grid(False)
    vals = assoc_m.to_numpy()
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i != j and vals[i, j] >= 0.3:
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if vals[i, j] > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Cramér's V (bias-corrected)")
    ax.set_title("Pairwise association (train); values ≥ 0.30 annotated")
    return _save(fig, "05_association_heatmap.png")


def fig_target_association(assoc: pd.DataFrame) -> str:
    """Horizontal bars: Cramér's V between each feature and the label (TRAIN)."""
    a = assoc.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(a["feature"], a["cramers_v"], color=LEGIT, height=0.65)
    for y, v in enumerate(a["cramers_v"]):
        if v >= 0.1:
            ax.text(v, y, f" {v:.2f}", va="center", fontsize=7, color=INK)
    ax.set_xlabel("Cramér's V with phishing label")
    ax.set_xlim(0, max(a["cramers_v"].max() * 1.15, 0.1))
    ax.grid(axis="y", visible=False)
    ax.set_title("Feature–label association (training partition)")
    return _save(fig, "06_feature_target_association.png")


def fig_single_feature_auc(auc: pd.DataFrame) -> str:
    """Single-feature validation ROC-AUC with 95% bootstrap intervals."""
    a = auc.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.errorbar(a["val_roc_auc"], range(len(a)),
                xerr=[a["val_roc_auc"] - a["ci_low"], a["ci_high"] - a["val_roc_auc"]],
                fmt="o", color=LEGIT, ecolor=INK_2, elinewidth=1, capsize=2, markersize=5)
    ax.axvline(0.5, color=INK_2, ls="--", lw=1)
    ax.text(0.5, len(a) - 0.4, " chance", color=INK_2, fontsize=8)
    for y, v in enumerate(a["val_roc_auc"]):
        if v >= 0.6:
            ax.text(a["ci_high"][y] + 0.005, y, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(a)), a["feature"])
    ax.set_xlabel("Validation ROC-AUC (fit on train)")
    ax.set_ylim(-0.7, len(a) + 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Single-feature validation ROC-AUC (95% bootstrap CI)")
    return _save(fig, "07_single_feature_val_auc.png")


def fig_indicator_count(train: pd.DataFrame, features: list[str]) -> str:
    """Distribution of the number of features coded -1 per row, by class (TRAIN)."""
    count = (train[features] == -1).sum(axis=1)
    bins = np.arange(count.min(), count.max() + 2) - 0.5
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for cls, color, name in [(1, PHISH, "Phishing"), (0, LEGIT, "Legitimate")]:
        c = count[train[TARGET] == cls]
        ax.hist(c, bins=bins, histtype="step", color=color, linewidth=2,
                label=f"{name} (median {int(c.median())})")
    ax.set_xlabel("Number of features coded −1 in the row")
    ax.set_ylabel("Training rows")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("How many '−1' indicators a site has, by class (train)")
    return _save(fig, "08_minus_one_count_by_class.png")


# ---------------------------------------------------------------------------- main


def run_all() -> dict[str, Any]:
    """Produce every Phase 2 figure and return (and save) the numeric results."""
    _style()
    d = load_eda_data()
    raw, cleaned, train, val, features = d["raw"], d["cleaned"], d["train"], d["val"], d["features"]
    audit = json.loads((REPORTS_DIR / "cleaning_log.json").read_text(encoding="utf-8"))["duplicate_audit"]

    assoc = target_association(train, features)
    assoc_m = association_matrix(train, features + [TARGET])
    auc = single_feature_auc(train, val, features)
    leak = duplicate_leakage_diagnostic(raw, features)
    upper = assoc_m.loc[features, features].where(np.triu(np.ones((len(features),) * 2, bool), 1))
    top_pairs = upper.stack().sort_values(ascending=False).head(10)

    figures = {
        "class_distribution": fig_class_distribution(raw, cleaned),
        "duplicates": fig_duplicates(raw, audit),
        "values_by_class": fig_stacked_by_class(train, features),
        "phishing_rate": fig_phishing_rate(assoc, float(train[TARGET].mean())),
        "association_heatmap": fig_association_heatmap(assoc_m),
        "target_association": fig_target_association(assoc),
        "single_feature_auc": fig_single_feature_auc(auc),
        "minus_one_count": fig_indicator_count(train, features),
    }
    minus_one = (train[features] == -1).sum(axis=1)
    results = {
        "partition_sizes": {"cleaned": len(cleaned), "train": len(d["split"].train),
                            "val": len(d["split"].val), "test": len(d["split"].test)},
        "partition_phishing_rate": {"train": round(float(train[TARGET].mean()), 4),
                                    "val": round(float(val[TARGET].mean()), 4)},
        "value_counts_cleaned": {f: {int(k): int(v) for k, v in cleaned[f].value_counts().sort_index().items()}
                                 for f in features},
        "target_association_train": assoc.to_dict(orient="records"),
        "single_feature_val_auc": auc.to_dict(orient="records"),
        "top_feature_pairs_cramers_v": [{"a": a, "b": b, "v": round(float(v), 4)}
                                        for (a, b), v in top_pairs.items()],
        "minus_one_count_median": {"phishing": float(minus_one[train[TARGET] == 1].median()),
                                   "legitimate": float(minus_one[train[TARGET] == 0].median())},
        "duplicate_leakage_diagnostic": leak,
        "figures": figures,
    }
    (REPORTS_DIR / "eda_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    r = run_all()
    print("Partitions:", r["partition_sizes"], "phishing rate:", r["partition_phishing_rate"])
    print("\nTop 10 single-feature validation ROC-AUC:")
    print(pd.DataFrame(r["single_feature_val_auc"]).head(10).to_string(index=False))
    print("\nBottom 5:")
    print(pd.DataFrame(r["single_feature_val_auc"]).tail(5).to_string(index=False))
    print("\nTop feature-label Cramér's V (train):")
    print(pd.DataFrame(r["target_association_train"])[["feature", "cramers_v", "mutual_info_bits",
                                                       "phishing_rate_by_value"]].head(10).to_string(index=False))
    print("\nStrongest feature pairs:", r["top_feature_pairs_cramers_v"][:6])
    print("\nMedian # of -1 codes:", r["minus_one_count_median"])
    print("\nDuplicate leakage diagnostic:", r["duplicate_leakage_diagnostic"])
    print("\nFigures:", *r["figures"].values(), sep="\n  ")
