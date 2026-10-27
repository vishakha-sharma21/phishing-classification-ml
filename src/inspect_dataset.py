"""Phase 1 — dataset inspection and duplicate/leakage audit (no model training).

Run with ``python -m src.inspect_dataset``. Writes:
    reports/dataset_inspection.md
    reports/cleaning_log.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    EXPECTED_FEATURE_VALUES,
    LABEL_MAPPING,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.data_loading import load_raw

# Features whose "phishing" condition (per the UCI feature document) describes a
# rare anomaly in a URL/page. Used only as a sanity check of the code direction.
RARE_ANOMALY_FEATURES = {
    "having_ip_address": "domain part is an IP address",
    "having_at_symbol": "URL contains '@'",
    "shortining_service": "URL uses a shortening service",
    "double_slash_redirecting": "'//' appears after the protocol",
    "port": "non-preferred port status",
    "submitting_to_email": "form submits via mail()/mailto:",
    "on_mouseover": "onMouseOver changes the status bar",
    "rightclick": "right-click is disabled",
    "iframe": "uses invisible iframe",
}


def sha256_of(path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def class_counts(y: pd.Series) -> dict[str, Any]:
    """Counts and percentages of the original labels."""
    counts = y.value_counts().sort_index()
    return {
        str(int(k)): {"count": int(v), "percent": round(100 * v / len(y), 2)}
        for k, v in counts.items()
    }


def audit_duplicates(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """Audit exact duplicates and feature-identical groups with conflicting labels."""
    n = len(df)
    exact_dup_rows = int(df.duplicated(keep="first").sum())
    unique_rows = df.drop_duplicates()

    grouped = df.groupby(features, sort=False)[TARGET_COLUMN]
    n_labels = grouped.nunique()
    group_size = grouped.size()
    n_unique_vectors = len(group_size)
    conflicting = n_labels[n_labels > 1]

    # Detail per conflicting group: how many raw rows carry each label.
    conflict_details = []
    conflict_keys = set(conflicting.index)
    for key, sub in df.groupby(features, sort=False):
        if key in conflict_keys:
            vc = sub[TARGET_COLUMN].value_counts()
            conflict_details.append(
                {str(int(k)): int(v) for k, v in vc.sort_index().items()}
            )
    conflict_raw_rows = int(group_size[conflicting.index].sum())

    # Policy: collapse exact duplicates; drop every vector that has both labels.
    vec_key = unique_rows[features].apply(tuple, axis=1)
    is_conflict = vec_key.isin(conflict_keys)
    cleaned = unique_rows[~is_conflict.values]

    dup_group_sizes = group_size[group_size > 1]
    return {
        "original_rows": n,
        "exact_duplicate_rows_beyond_first": exact_dup_rows,
        "unique_rows_features_plus_label": len(unique_rows),
        "unique_feature_vectors": n_unique_vectors,
        "feature_vectors_appearing_more_than_once": int(len(dup_group_sizes)),
        "largest_feature_vector_group_size": int(group_size.max()),
        "conflicting_feature_vector_groups": int(len(conflicting)),
        "raw_rows_in_conflicting_groups": conflict_raw_rows,
        "unique_rows_in_conflicting_groups": int(is_conflict.sum()),
        "conflict_label_counts_per_group": conflict_details,
        "retained_rows_after_policy": int(len(cleaned)),
        "class_counts_original": class_counts(df[TARGET_COLUMN]),
        "class_counts_after_exact_dedup": class_counts(unique_rows[TARGET_COLUMN]),
        "class_counts_after_policy": class_counts(cleaned[TARGET_COLUMN]),
        "_cleaned": cleaned,
    }


def feature_label_association(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Descriptive per-feature association with the original label (full data).

    ``best_rule_agreement`` is the accuracy of the best per-value majority rule,
    i.e. how well a feature alone separates the labels. It is computed on all rows
    and is descriptive only; validation-based diagnostics come in later phases.
    """
    y = df[TARGET_COLUMN]
    rows = []
    for f in features:
        x = df[f]
        tab = pd.crosstab(x, y)
        best = tab.max(axis=1).sum() / len(df)
        neg1_rate = {
            int(v): round(float((y[x == v] == -1).mean()), 4) for v in sorted(x.unique())
        }
        rows.append(
            {
                "feature": f,
                "equals_target": bool((x == y).all()),
                "equals_negated_target": bool((x == -y).all()),
                "pearson_r_with_target": round(float(np.corrcoef(x, y)[0, 1]), 4),
                "best_rule_agreement": round(float(best), 4),
                "label_-1_rate_by_value": neg1_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("best_rule_agreement", ascending=False)


def direction_check(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check which code is the rare one for anomaly-type features and how it relates
    to the label. If code -1 is rare (the anomaly) and is associated with label -1,
    that supports "-1 = phishing" for both features and target."""
    y = df[TARGET_COLUMN]
    out = []
    for f, meaning in RARE_ANOMALY_FEATURES.items():
        vc = df[f].value_counts()
        rare_code = int(vc.idxmin())
        out.append(
            {
                "feature": f,
                "anomaly": meaning,
                "value_counts": {str(int(k)): int(v) for k, v in vc.sort_index().items()},
                "rare_code": rare_code,
                "label_-1_rate_when_rare": round(float((y[df[f] == rare_code] == -1).mean()), 4),
                "label_-1_rate_otherwise": round(float((y[df[f] != rare_code] == -1).mean()), 4),
            }
        )
    return out


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    df = load_raw()
    features = [c for c in df.columns if c != TARGET_COLUMN]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Structure, types, values ---------------------------------------------------
    print(f"File: {RAW_DATA_PATH}  sha256={sha256_of(RAW_DATA_PATH)}")
    print(f"Shape: {df.shape}  features={len(features)}  target='{TARGET_COLUMN}'")
    dtypes = df.dtypes.astype(str).value_counts().to_dict()
    print(f"Dtypes: {dtypes}")
    missing = int(df.isna().sum().sum())
    print(f"Missing values: {missing}")

    value_rows = []
    invalid = {}
    for c in df.columns:
        vals = sorted(int(v) for v in df[c].unique())
        bad = sorted(set(vals) - EXPECTED_FEATURE_VALUES)
        if bad:
            invalid[c] = bad
        counts = df[c].value_counts().sort_index()
        value_rows.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "values": vals,
                "counts": ", ".join(f"{int(k)}:{int(v)}" for k, v in counts.items()),
            }
        )
    values_df = pd.DataFrame(value_rows)
    n_binary = sum(len(r["values"]) == 2 for r in value_rows if r["column"] != TARGET_COLUMN)
    n_ternary = sum(len(r["values"]) == 3 for r in value_rows if r["column"] != TARGET_COLUMN)
    print(f"Binary features: {n_binary}, ternary features: {n_ternary}")
    print(f"Values outside {{-1,0,1}}: {invalid or 'none'}")
    zero_one = [r["column"] for r in value_rows if r["values"] == [0, 1]]
    print(f"Columns coded {{0,1}} instead of {{-1,1}}: {zero_one}")

    target_vals = sorted(int(v) for v in df[TARGET_COLUMN].unique())
    print(f"Target values: {target_vals}")
    print(f"Class distribution (original labels): {class_counts(df[TARGET_COLUMN])}")

    # --- Duplicates ------------------------------------------------------------------
    audit = audit_duplicates(df, features)
    cleaned = audit.pop("_cleaned")
    print("\nDuplicate audit:")
    for k, v in audit.items():
        if k != "conflict_label_counts_per_group":
            print(f"  {k}: {v}")
    print(f"  conflict group label counts (first 10): {audit['conflict_label_counts_per_group'][:10]}")

    # --- Leakage / suspicious features -----------------------------------------------
    assoc = feature_label_association(df, features)
    target_like = assoc[assoc.equals_target | assoc.equals_negated_target].feature.tolist()
    strong = assoc[assoc.best_rule_agreement >= 0.85].feature.tolist()
    print(f"\nColumns identical to target or its negation: {target_like or 'none'}")
    print("Top 8 features by single-feature majority-rule agreement (all rows, descriptive):")
    print(assoc.head(8)[["feature", "pearson_r_with_target", "best_rule_agreement"]].to_string(index=False))

    # Same statistic after cleaning, to see whether duplicates inflate it.
    assoc_clean = feature_label_association(cleaned, features).set_index("feature")
    assoc = assoc.assign(
        best_rule_agreement_cleaned=assoc.feature.map(assoc_clean.best_rule_agreement)
    )

    direction = direction_check(df)
    rare_is_neg1 = sum(d["rare_code"] == -1 for d in direction)
    supports = sum(
        d["rare_code"] == -1 and d["label_-1_rate_when_rare"] > d["label_-1_rate_otherwise"]
        for d in direction
    )
    r = assoc.set_index("feature").pearson_r_with_target
    n_pos_r, n_neg_r = int((r > 0).sum()), int((r < 0).sum())
    neg_r_features = r[r < 0].sort_values().round(4).to_dict()
    print(f"\nDirection check: rare code is -1 for {rare_is_neg1}/{len(direction)} anomaly features; "
          f"{supports}/{len(direction)} also show a higher label -1 rate when the anomaly is present")
    print(f"Features positively / negatively correlated with target: {n_pos_r} / {n_neg_r}; "
          f"negative: {neg_r_features}")
    for d in direction:
        print(f"  {d['feature']:<26} counts={d['value_counts']} rare={d['rare_code']} "
              f"P(-1|rare)={d['label_-1_rate_when_rare']} P(-1|other)={d['label_-1_rate_otherwise']}")

    # --- Write cleaning log --------------------------------------------------------
    log = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "dataset": "UCI Phishing Websites (id=327)",
            "doi": "10.24432/C51W2X",
            "license": "CC BY 4.0",
            "file": str(RAW_DATA_PATH.relative_to(RAW_DATA_PATH.parents[2])),
            "sha256": sha256_of(RAW_DATA_PATH),
        },
        "shape": list(df.shape),
        "target_column": TARGET_COLUMN,
        "target_values_observed": target_vals,
        "label_mapping_original_to_project": {str(k): v for k, v in LABEL_MAPPING.items()},
        "label_mapping_meaning": {"1": "phishing (positive)", "0": "legitimate"},
        "missing_values": missing,
        "values_outside_expected_set": invalid,
        "columns_coded_zero_one": zero_one,
        "duplicate_audit": audit,
        "policy": {
            "exact_duplicates": "collapse to a single row (dataset treated as a set of unique feature vectors)",
            "conflicting_groups": "exclude every row of a feature vector observed with both labels; "
                                  "no label is chosen arbitrarily",
            "order": "deduplicate before train/validation/test splitting",
            "status": "APPROVED by user (Phase 1 review)",
        },
        "target_like_columns": target_like,
        "features_with_rule_agreement_ge_0.85": strong,
        "direction_check": direction,
    }
    (REPORTS_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    # --- Write markdown report -------------------------------------------------------
    a = audit
    cc_o, cc_d, cc_p = (a["class_counts_original"], a["class_counts_after_exact_dedup"],
                        a["class_counts_after_policy"])

    def fmt_cc(cc):
        return ", ".join(f"`{k}`: {v['count']} ({v['percent']}%)" for k, v in cc.items())

    assoc_md = assoc[["feature", "pearson_r_with_target", "best_rule_agreement",
                      "best_rule_agreement_cleaned"]]
    dir_md = pd.DataFrame(direction)[["feature", "anomaly", "value_counts", "rare_code",
                                      "label_-1_rate_when_rare", "label_-1_rate_otherwise"]]
    md = f"""# Dataset Inspection — UCI Phishing Websites

*Generated by `python -m src.inspect_dataset` on {log['generated_at_utc']}. All numbers come from executed code.*

## 1. Source and provenance

| Item | Value |
|---|---|
| Dataset | UCI Phishing Websites, id 327 |
| URL | https://archive.ics.uci.edu/dataset/327/phishing+websites |
| DOI | https://doi.org/10.24432/C51W2X |
| Creators | Rami Mohammad, Lee McCluskey (UCI metadata); paper also lists Fadi Thabtah |
| Intro paper | Mohammad, Thabtah, McCluskey (2012), *An assessment of features related to phishing websites using an automated technique*, ICITST |
| License | CC BY 4.0 (stated on the UCI dataset page) |
| Local file | `data/raw/phishing.csv` |
| SHA-256 | `{log['source']['sha256']}` |

**How it was obtained.** `fetch_ucirepo(id=327)` retrieved the metadata but failed to download
`data.csv` with `SSLCertVerificationError: certificate has expired`. The UCI server certificate was
checked and is valid (Let's Encrypt, valid 2026-08-21 → 2026-11-19), and `curl` with normal verification
succeeded, so the fault is in the local Python certificate store. The same `data_url` that ucimlrepo uses
(`https://archive.ics.uci.edu/static/public/327/data.csv`) was downloaded with `curl`.
TLS verification was **not** disabled.

**Cross-check.** The official zip (`phishing+websites.zip`) was also downloaded. Its
`Training Dataset.arff` has identical values (11,055 × 31, every cell equal) to `data.csv`. The CSV uses
lowercase snake_case names and corrects two ARFF typos (`Domain_registeration_length` →
`domain_registration_length`, `popUpWidnow` → `popupwindow`). The zip also contains `.old.arff`
(`@relation test`, a different `{{1,0}}` coding, ~2.5k lines), which is **not used**. The feature description
document is saved at `docs/source/`, and the API metadata at `docs/source/uci_metadata.json`.

## 2. Structure

- Shape: **{df.shape[0]} rows × {df.shape[1]} columns** ({len(features)} features + target `{TARGET_COLUMN}`).
- Dtypes: {dtypes}
- Missing values: **{missing}**
- Values outside {{-1, 0, 1}}: **{invalid or 'none'}**
- Binary features: {n_binary}; ternary features: {n_ternary}.
- Columns coded {{0, 1}} rather than {{-1, 1}}: **{zero_one}**. The ARFF header also declares
  `Redirect {{0,1}}`. The feature document describes three redirect levels (≤1 legitimate, 2–3 suspicious,
  ≥4 phishing), but only two codes occur. Which levels 0 and 1 represent is undocumented, so this is
  recorded as an ambiguity.

### Per-column values and counts

{md_table(values_df)}

## 3. Target and label mapping

- Observed target values: {target_vals}
- Original class distribution: {fmt_cc(cc_o)}
- Project convention: **1 = phishing (positive)**, **0 = legitimate**. Mapping: original `-1 → 1`, `+1 → 0`.

**Evidence for the direction.** Neither the UCI metadata nor the feature document states which code means
phishing, for the features or for the target. The variable `result` has no description. "-1 = phishing" is
the convention used for this dataset in the literature and in this project's specification. Two internal checks
were run:

*Step 1 — feature code direction.* For each feature below, the "phishing" condition in the feature document is
an anomaly (an IP address in the URL, an '@' in the URL, and so on) that should be the less frequent case.

{md_table(dir_md)}

The rare code is `-1` for **{rare_is_neg1}/{len(direction)}** of these features. This supports
"feature code -1 = phishing indicator". However, these features are only weakly related to the label: just
**{supports}/{len(direction)}** show a higher target `-1` rate when the anomaly is present, and the differences are small.

*Step 2 — target direction.* **{n_pos_r}** features correlate positively with the target and **{n_neg_r}**
negatively. The two strongest predictors, `sslfinal_state` (r = {r['sslfinal_state']}) and `url_of_anchor`
(r = {r['url_of_anchor']}), are strongly positive. Their phishing-indicator code `-1` co-occurs with target `-1`.
Combined with step 1, this supports **target -1 = phishing**.

This is a plausibility argument, not documentary proof. It assumes the documented anomalies really are rarer
than their absence in the collected URLs.

**Features that run against their documented rule** (negative r: the "phishing" code co-occurs with the
legitimate label): {neg_r_features}. Most are ~0 and probably just noise. `domain_registration_length`
(r = {r['domain_registration_length']}) is a clear exception. Its documented rule says "expires in ≤ 1 year →
phishing", yet code `-1` is associated with *legitimate* sites here. Possible explanations include a coding
inversion in the source, or a genuine dataset-specific pattern. This cannot be resolved from the available
documentation and will be flagged in `reports/features.md`.

## 4. Duplicate audit

| Quantity | Value |
|---|---|
| Original rows | {a['original_rows']} |
| Exact duplicate rows (features + label, beyond first occurrence) | {a['exact_duplicate_rows_beyond_first']} |
| Unique rows (features + label) | {a['unique_rows_features_plus_label']} |
| Unique feature vectors | {a['unique_feature_vectors']} |
| Feature vectors appearing more than once | {a['feature_vectors_appearing_more_than_once']} |
| Largest group of identical feature vectors | {a['largest_feature_vector_group_size']} |
| Feature vectors with conflicting labels | {a['conflicting_feature_vector_groups']} |
| Raw rows inside conflicting groups | {a['raw_rows_in_conflicting_groups']} |
| Unique rows inside conflicting groups | {a['unique_rows_in_conflicting_groups']} |
| **Rows retained under proposed policy** | **{a['retained_rows_after_policy']}** |

Class counts:
- Original: {fmt_cc(cc_o)}
- After collapsing exact duplicates: {fmt_cc(cc_d)}
- After proposed policy: {fmt_cc(cc_p)}

**Interpretation.** The 30 features are coarse discretised indicators (2–3 levels each). Distinct websites can
therefore legitimately produce the same feature vector, so a duplicate is not necessarily a copying error.
It may be a repeated observation of the same site, or two different sites that look identical at this
resolution. The evaluation problem is that identical rows can land in both the training and test sets.
A memorising model (such as a deep tree) then gets credit for recall rather than generalisation.

Conflicting groups are feature vectors labelled both phishing and legitimate. These sites cannot be
separated using these 30 features. They are irreducible error at this feature resolution, not something a
model should "fix".

**Policy (approved).**
1. Collapse exact feature+label duplicates to one row. This turns the population into *unique feature vectors*
   rather than *collected websites*, which changes the evaluated population. Frequent vectors lose their weight.
2. Exclude all rows of conflicting feature vectors. No label is chosen arbitrarily. Their count is recorded above,
   and the impact will be shown as a diagnostic.
3. Deduplicate **before** the 70/15/15 stratified split.
4. As a documented diagnostic only, report results on the raw (non-deduplicated) data as well, to show how much
   duplicates inflate the scores.

## 5. Leakage screening

- Columns identical to the target or its negation: **{target_like or 'none'}**.
- Single-feature association is descriptive and computed on all rows; no models were trained.
  `best_rule_agreement` is the accuracy of predicting the majority label for each value of a single feature.
  The table also shows the same statistic after the proposed cleaning.

{md_table(assoc_md)}

Features with agreement ≥ 0.85: **{strong or 'none'}**. These need validation-based single-feature checks in
Phase 2, interpreted with the feature document. A strong single predictor is a warning to investigate,
not proof of leakage.

## 6. Decisions (approved after Phase 1 review)

1. Approve the duplicate policy above (collapse exact duplicates, drop conflicting vectors, then split)?
2. Accept the `-1 → phishing (1)`, `+1 → legitimate (0)` mapping on the evidence in §3?
3. Note that `redirect` is coded {{0, 1}} with undocumented level meaning.
"""
    (REPORTS_DIR / "dataset_inspection.md").write_text(md, encoding="utf-8")
    print(f"\nWrote {REPORTS_DIR / 'dataset_inspection.md'} and {REPORTS_DIR / 'cleaning_log.json'}")


if __name__ == "__main__":
    main()
