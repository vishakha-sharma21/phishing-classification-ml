"""Logistic-Regression-only versions of the evaluation, overfitting and interpretation figures.

Run with ``python -m src.lr_only_figures`` after ``python -m src.train``. Reads the saved
model and reports/results.json; the test set is only scored for descriptive plots of the
already-frozen model (no choices are made here).
"""

from __future__ import annotations

from typing import Any

from src import plotting
from src.config import MODELS_DIR
from src.logistic_regression import LogisticRegression
from src.reference_models import SingleFeatureBaseline, sk_proba, sklearn_logistic
from src.train import get_data, load_results, matrices


def main() -> dict[str, Any]:
    plotting.style()
    res = load_results()
    ev, sel = res["evaluation"], res["logistic_regression"]["selected"]
    data, enc = get_data()
    lr_m, raw = matrices(data, enc[sel["encoding"]]), matrices(data, enc["raw"])
    (Xtr, ytr), (Xte, yte) = lr_m["train"], lr_m["test"]
    best = res["baselines_val"]["best_single_feature"]["feature"]

    model = LogisticRegression.load(MODELS_DIR / "logistic_regression.json")
    probas = {
        "Custom LR": model.predict_proba(Xte),
        "sklearn LR (ref)": sk_proba(sklearn_logistic(sel["l2"], len(ytr)).fit(Xtr, ytr), Xte),
        "Best single feature": SingleFeatureBaseline(data.features.index(best)).fit(raw["train"][0], ytr)
                                .predict_proba(raw["test"][0]),
    }
    figs = {
        "roc_pr": plotting.roc_pr_curves(yte, probas, "lr_05_roc_pr_test.png", "test set"),
        "confusion": plotting.confusion_grid({k: ev["test"][k] for k in ("Custom LR", "Best single feature")},
                                             "lr_06_confusion_test.png", "Confusion matrices — test set (rows: actual)"),
    }

    perm = {r["feature"]: r["mean_auc_drop"] for r in res["interpretation"]["permutation_importance_val"]["Custom LR"]}
    top = dict(sorted(perm.items(), key=lambda kv: -kv[1])[:12])
    figs["permutation"] = plotting.bar_h(top, "Mean ROC-AUC drop when shuffled (10 repeats, validation)",
                                         "Logistic Regression: permutation importance (top 12)",
                                         "lr_07_permutation_val.png")

    lc = res["overfitting_feature_selection"]["learning_curves"]["Custom LR"]
    figs["learning_curve"] = plotting.line_panel(
        {"train": (lc["n"], lc["train_roc_auc"], "--", plotting.SERIES[0]),
         "validation": (lc["n"], lc["val_roc_auc"], "-", plotting.SERIES[0])},
        "Training rows used", "ROC-AUC", "Logistic Regression learning curve", "lr_08_learning_curve.png")

    fs = res["overfitting_feature_selection"]["feature_selection_val"]
    ks = [3, 5, 8, 10, 15, 20, 30]
    rows = [fs[f"mi_top_{k}"] if k < 30 else fs["all_features"] for k in ks]
    figs["mi_top_k"] = plotting.line_panel(
        {"Custom LR": (ks, [r["lr_val"]["roc_auc"] for r in rows])},
        "Top-k features by training mutual information", "Validation ROC-AUC",
        "Feature selection: validation ROC-AUC vs number of features", "lr_09_mi_top_k.png")
    figs["ablation"] = plotting.bar_h(
        {k.replace("ablation_", ""): fs[k]["lr_val"]["roc_auc"] - fs["all_features"]["lr_val"]["roc_auc"]
         for k in fs if k.startswith("ablation_") or k.startswith("near_constant")},
        "Change in LR validation ROC-AUC vs all 30 features", "Ablations and filters (Logistic Regression, validation)",
        "lr_10_ablation.png", signed=True)
    return figs


if __name__ == "__main__":
    for k, v in main().items():
        print(f"{k:<15} {v}")
