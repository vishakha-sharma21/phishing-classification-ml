"""Generate reports/features.md from src/feature_info.py and reports/eda_results.json.

Run with ``python -m src.feature_docs`` (after ``python -m src.eda``).
"""

from __future__ import annotations

import json

from src.config import REPORTS_DIR
from src.feature_info import FEATURES


def main() -> None:
    eda = json.loads((REPORTS_DIR / "eda_results.json").read_text(encoding="utf-8"))
    assoc = {r["feature"]: r for r in eda["target_association_train"]}
    auc = {r["feature"]: r for r in eda["single_feature_val_auc"]}
    counts = eda["value_counts_cleaned"]

    lines = [
        "# Feature descriptions — UCI Phishing Websites",
        "",
        "Source: the feature document distributed with the dataset (`docs/source/Phishing Websites Features.docx`, "
        "text in `docs/source/uci_features_description.txt`), Mohammad, Thabtah & McCluskey; dataset licensed CC BY 4.0.",
        "",
        "**How to read the codes.** The documentation describes each indicator as *phishing / suspicious / legitimate* but "
        "never states the numeric codes. This project uses the conventional reading −1 = phishing-like, 0 = suspicious, "
        "1 = legitimate-like, which the Phase 1 checks support (9/9 anomaly features have −1 as the rare code) but do not prove. "
        "Observed statistics below come from executed code: counts use all 5,721 cleaned rows; phishing rates and Cramér's V "
        "use the training partition only; ROC-AUC is a one-feature model fitted on train and scored on validation. "
        "All are associations, not causal effects.",
        "",
        "## Summary",
        "",
        "| Feature | Group | Levels | Cramér's V (train) | 1-feature val ROC-AUC |",
        "|---|---|---|---|---|",
    ]
    for f, info in FEATURES.items():
        lv = ", ".join(str(k) for k in counts[f])
        lines.append(f"| `{f}` | {info['group']} | {lv} | {assoc[f]['cramers_v']:.3f} | {auc[f]['val_roc_auc']:.3f} |")
    lines += ["", "## Details", ""]
    for f, info in FEATURES.items():
        rates = assoc[f]["phishing_rate_by_value"]
        lines += [
            f"### `{f}` — {info['label']}",
            "",
            f"* **Group:** {info['group']}",
            f"* **Documented rule:** {info['rule']}",
            "* **Values (conventional reading → observed):**",
        ]
        for code, meaning in info["values"].items():
            n = counts[f].get(str(code), 0)
            r = rates.get(str(code))
            lines.append(f"  * `{code}` — {meaning}: {n} cleaned rows" +
                         (f"; phishing rate in train {r:.1%}" if r is not None else ""))
        lines.append(f"* **Single-feature validation ROC-AUC:** {auc[f]['val_roc_auc']:.3f} "
                     f"[{auc[f]['ci_low']:.3f}, {auc[f]['ci_high']:.3f}]")
        if info.get("note"):
            lines.append(f"* **Limitations / ambiguity:** {info['note']}")
        lines.append("")
    lines += [
        "## General limitations",
        "",
        "* All 30 indicators are pre-discretised by the dataset authors; the thresholds (e.g. 54/75 characters, 22%/61%) "
        "were chosen on their own data and cannot be re-derived here.",
        "* Several indicators depend on external services as of c. 2012 (WHOIS, Alexa rank, Google PageRank, PhishTank / "
        "StopBadware top lists). Some no longer exist, so the features cannot be reproduced exactly for current websites.",
        "* Strong co-occurrence between unrelated checks (e.g. `favicon`–`popupwindow`, Cramér's V 0.93) suggests that "
        "the extraction or collection process shaped the data; this cannot be verified from the published files.",
        "",
    ]
    (REPORTS_DIR / "features.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORTS_DIR / 'features.md'}")


if __name__ == "__main__":
    main()
