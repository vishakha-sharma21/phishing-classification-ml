"""End-to-end training and evaluation pipeline.

    python -m src.train                 # every stage, in order
    python -m src.train --stage NAME    # one stage (earlier stages' outputs must exist)

Each stage merges its results into reports/results.json under its own key, so every
reported number is traceable to the stage that produced it. Model selection uses the
validation partition only; the test partition is read only in the ``evaluate`` stage.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from src.config import LABEL_MAPPING, MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR
from src.data_loading import load_raw
from src import plotting
from src.cross_validation import cross_validate, paired_fold_difference, stratified_folds, summarise
from src.feature_selection import dominant_level_share, mutual_information_ranking, near_constant_filter
from src.evaluation import bootstrap_ci, compute_metrics, paired_bootstrap_diff, verdict
from src.logistic_regression import LogisticRegression
from src.preprocessing import TARGET, FeatureEncoder, PreparedData, prepare, save_split
from src.decision_tree import DecisionTree
from src.interpretation import permutation_importance, rank_stability
from src.random_forest import RandomForest
from src.reference_models import (MajorityBaseline, SingleFeatureBaseline, sk_proba, sklearn_forest,
                                   sklearn_logistic, sklearn_tree)
from src.threshold_tuning import threshold_curve, tune_threshold

RESULTS_PATH = REPORTS_DIR / "results.json"


# ------------------------------------------------------------------ helpers


def _round(obj: Any, nd: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, np.generic):
        return _round(obj.item(), nd)
    return obj


def load_results() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}


def update_results(key: str, value: Any) -> None:
    """Merge one stage's output into reports/results.json."""
    res = load_results()
    res[key] = _round(value)
    RESULTS_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")


def get_data() -> tuple[PreparedData, dict[str, FeatureEncoder]]:
    """Cleaned data, the approved split, and encoders fitted on the training partition."""
    data = prepare(load_raw())
    train = data.part("train")
    encoders = {k: FeatureEncoder(k).fit(train) for k in ("raw", "onehot")}
    return data, encoders


def matrices(data: PreparedData, enc: FeatureEncoder) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {p: (enc.transform(data.part(p)), data.y(p)) for p in ("train", "val", "test")}


# ------------------------------------------------------------- stage: prepare


def stage_prepare() -> dict[str, Any]:
    """Phase 3: write cleaned data, split indices and the preprocessing config."""
    data, encoders = get_data()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.cleaned.to_csv(PROCESSED_DIR / "cleaned.csv", index=False)
    save_split(data, PROCESSED_DIR / "splits.json")

    # Every level seen in val/test must have been seen in train (else encoding would fail).
    unseen = {f: sorted(set(data.cleaned[f]) - set(encoders["raw"].levels[f])) for f in data.features}
    unseen = {f: v for f, v in unseen.items() if v}
    if unseen:
        raise ValueError(f"levels absent from training data: {unseen}")

    config = {
        "label_mapping_original_to_project": {str(k): v for k, v in LABEL_MAPPING.items()},
        "positive_class": "phishing (1)",
        "feature_order": data.features,
        "allowed_values": encoders["raw"].levels,
        "encoders": {k: e.to_dict() for k, e in encoders.items()},
        "cleaning_policy": "collapse exact duplicates; drop feature vectors seen with both labels; then split",
        "split": {"seed": RANDOM_SEED, "train": len(data.split.train), "val": len(data.split.val),
                  "test": len(data.split.test), "stratified": True},
        "scaling": "none (bounded discrete codes; one-hot indicators are already on a 0/1 scale)",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "preprocessing.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        "cleaned_rows": len(data.cleaned),
        "partition_sizes": config["split"],
        "phishing_rate": {p: float(data.y(p).mean()) for p in ("train", "val", "test")},
        "encoded_dimensions": {k: len(e.output_names) for k, e in encoders.items()},
    }
    update_results("preprocessing", summary)
    return summary


# ----------------------------------------------------------- stage: baselines


def stage_baselines() -> dict[str, Any]:
    """Phase 3 baselines on VALIDATION (threshold 0.5): majority, best single feature,
    and the scikit-learn logistic regression REFERENCE under both encodings."""
    data, enc = get_data()
    raw = matrices(data, enc["raw"])
    (Xtr, ytr), (Xva, yva) = raw["train"], raw["val"]
    out: dict[str, Any] = {}

    maj = MajorityBaseline().fit(Xtr, ytr)
    out["majority"] = compute_metrics(yva, maj.predict_proba(Xva))

    # Best single feature chosen by validation ROC-AUC of the train-fitted rate model.
    aucs = {}
    for j, f in enumerate(data.features):
        p = SingleFeatureBaseline(j).fit(Xtr, ytr).predict_proba(Xva)
        aucs[f] = compute_metrics(yva, p)["roc_auc"]
    best = max(aucs, key=aucs.get)
    sfb = SingleFeatureBaseline(data.features.index(best)).fit(Xtr, ytr)
    out["best_single_feature"] = {"feature": best, **compute_metrics(yva, sfb.predict_proba(Xva))}

    for kind in ("raw", "onehot"):
        m = matrices(data, enc[kind])
        t0 = time.perf_counter()
        sk = sklearn_logistic(l2=0.0, n_train=len(ytr)).fit(m["train"][0], ytr)
        fit_s = time.perf_counter() - t0
        out[f"sklearn_lr_reference_{kind}"] = {
            **compute_metrics(yva, sk_proba(sk, m["val"][0])),
            "train_accuracy": compute_metrics(ytr, sk_proba(sk, m["train"][0]))["accuracy"],
            "n_inputs": m["train"][0].shape[1], "fit_seconds": fit_s,
        }
    out["_note"] = ("Validation partition, threshold 0.5. sklearn LR is an unpenalised REFERENCE model, "
                    "not the custom implementation.")
    update_results("baselines_val", out)
    return out


# ---------------------------------------------------- stage: logistic regression

# Learning rates are set relative to the smoothness constant L of each encoding:
# descent is guaranteed for lr < 2/L, so the grid probes 0.5/L, 1/L and 1.9/L.
LR_GRID = {"encoding": ["raw", "onehot"], "l2": [0.0, 1e-4, 1e-3, 1e-2, 1e-1], "lr_times_L": [0.5, 1.0, 1.9]}
LR_MAX_ITER = 30000
LR_TOL = 1e-9


def _fit_lr(X, y, Xv=None, yv=None, **params) -> tuple[LogisticRegression, float, float]:
    """Fit and return (model, seconds, peak traced memory in MB)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    m = LogisticRegression(**params).fit(X, y, Xv, yv)
    secs = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return m, secs, peak


def stage_lr() -> dict[str, Any]:
    """Phase 4: tune the custom logistic regression on VALIDATION and save it."""
    plotting.style()
    data, enc = get_data()
    mats = {k: matrices(data, e) for k, e in enc.items()}

    # 1. Grid over encoding x L2 x learning rate; select by validation log-loss.
    grid, smooth = [], {}
    for kind in LR_GRID["encoding"]:
        (Xtr, ytr), (Xva, yva) = mats[kind]["train"], mats[kind]["val"]
        L = LogisticRegression.smoothness_constant(Xtr)
        smooth[kind] = L
        for l2 in LR_GRID["l2"]:
            for frac in LR_GRID["lr_times_L"]:
                lr = round(frac / L, 4)
                try:
                    m, secs, _ = _fit_lr(Xtr, ytr, learning_rate=lr, l2=l2, n_iter=LR_MAX_ITER, tol=LR_TOL)
                except FloatingPointError:
                    grid.append({"encoding": kind, "l2": l2, "lr_times_L": frac, "learning_rate": lr,
                                 "diverged": True})
                    continue
                pv, pt = m.predict_proba(Xva), m.predict_proba(Xtr)
                grid.append({"encoding": kind, "l2": l2, "lr_times_L": frac, "learning_rate": lr, "diverged": False,
                             "iterations": m.n_iter_run, "converged": m.converged, "fit_seconds": secs,
                             "train_log_loss": compute_metrics(ytr, pt)["log_loss"],
                             "val_log_loss": compute_metrics(yva, pv)["log_loss"],
                             "val_roc_auc": compute_metrics(yva, pv)["roc_auc"],
                             "val_f1_at_0.5": compute_metrics(yva, pv)["f1"]})
    ok = [g for g in grid if not g["diverged"]]
    best = min(ok, key=lambda g: (round(g["val_log_loss"], 4), -g["l2"], g["learning_rate"]))
    kind, l2, lr = best["encoding"], best["l2"], best["learning_rate"]
    (Xtr, ytr), (Xva, yva) = mats[kind]["train"], mats[kind]["val"]

    # 2. Final model with validation-loss tracking.
    model, fit_s, peak_mb = _fit_lr(Xtr, ytr, Xva, yva, learning_rate=lr, l2=l2, n_iter=LR_MAX_ITER, tol=LR_TOL)
    pv = model.predict_proba(Xva)
    threshold = tune_threshold(yva, pv, "f1")
    t0 = time.perf_counter()
    for _ in range(100):
        model.predict_proba(Xva)
    infer_ms = (time.perf_counter() - t0) / 100 * 1000

    figs = {}
    h = model.history
    figs["loss_curve"] = plotting.line_panel(
        {"train objective": (range(1, len(h["train_loss"]) + 1), h["train_loss"]),
         "validation BCE": (range(1, len(h["val_loss"]) + 1), h["val_loss"])},
        "Iteration", "Loss", f"Custom LR loss ({kind}, λ={l2:g}, lr={lr:g})", "lr_01_train_val_loss.png", logy=True)

    # 3. Learning-rate study (same encoding / L2), fixed 2,000 iterations, no early stop.
    #    Rates span both sides of the 2/L stability bound.
    lr_curves, lr_summary = {}, {}
    two_over_L = 2 / LogisticRegression.smoothness_constant(Xtr, l2)
    for rate in (0.01, 0.1, 0.5, 1.0, 2.0, 4.0):
        try:
            m = LogisticRegression(learning_rate=rate, l2=l2, n_iter=2000, tol=0).fit(Xtr, ytr)
            lr_curves[f"lr={rate:g}"] = (range(1, 2001), m.history["train_loss"])
            hist = np.array(m.history["train_loss"])
            lr_summary[str(rate)] = {"final_train_objective": float(hist[-1]),
                                     "monotone": bool(np.all(np.diff(hist) <= 1e-12)),
                                     "val_log_loss": compute_metrics(yva, m.predict_proba(Xva))["log_loss"]}
        except FloatingPointError:
            lr_summary[str(rate)] = {"diverged": True}
    figs["learning_rates"] = plotting.line_panel(lr_curves, "Iteration", "Training objective",
                                                 f"Effect of learning rate (2/L = {two_over_L:.2f})",
                                                 "lr_02_learning_rates.png", logy=True)

    # 4. L2 path: train vs validation log-loss for each encoding (best lr per point).
    l2_series = {}
    for ci, k in enumerate(LR_GRID["encoding"]):
        pts = [min((g for g in ok if g["encoding"] == k and g["l2"] == lam), key=lambda g: g["val_log_loss"])
               for lam in LR_GRID["l2"]]
        l2_series[f"{k} – train"] = ([p["l2"] for p in pts], [p["train_log_loss"] for p in pts], "--",
                                     plotting.SERIES[ci])
        l2_series[f"{k} – validation"] = ([p["l2"] for p in pts], [p["val_log_loss"] for p in pts], "-",
                                          plotting.SERIES[ci])
    figs["l2_path"] = plotting.line_panel(l2_series, "L2 strength λ (symlog)", "Log-loss",
                                          "Regularisation and encoding: train vs validation", "lr_03_l2_encoding.png",
                                          logx=True)

    # 5. Class weighting and mini-batch variants (diagnostics, not selected automatically).
    bal = LogisticRegression(learning_rate=lr, l2=l2, n_iter=LR_MAX_ITER, tol=LR_TOL,
                             class_weight="balanced").fit(Xtr, ytr)
    t0 = time.perf_counter()
    mini = LogisticRegression(learning_rate=0.1, l2=l2, n_iter=100, batch_size=64, seed=RANDOM_SEED).fit(Xtr, ytr)
    mini_s = time.perf_counter() - t0
    variants = {
        "selected_unweighted": compute_metrics(yva, pv, 0.5),
        "balanced_class_weight": compute_metrics(yva, bal.predict_proba(Xva), 0.5),
        "minibatch_64_100_epochs": {**compute_metrics(yva, mini.predict_proba(Xva), 0.5), "fit_seconds": mini_s,
                                    "final_train_objective": mini.history["train_loss"][-1]},
    }

    # 6. Threshold study on validation.
    curve = threshold_curve(yva, pv)
    figs["threshold"] = plotting.threshold_plot(curve, threshold, "lr_04_threshold.png",
                                                "Custom LR: validation precision / recall / F1 vs threshold")

    # 7. Independent check against a scikit-learn REFERENCE with the same objective.
    sk = sklearn_logistic(l2, len(ytr)).fit(Xtr, ytr)
    psk = sk_proba(sk, Xva)
    reference = {
        "max_abs_weight_diff": float(np.max(np.abs(sk.coef_[0] - model.w))),
        "bias_diff": float(sk.intercept_[0] - model.b),
        "max_abs_val_proba_diff": float(np.max(np.abs(psk - pv))),
        "val_label_agreement_at_threshold": float(np.mean((psk >= threshold) == (pv >= threshold))),
        "sklearn_val_metrics": compute_metrics(yva, psk, threshold),
    }

    enc_obj = enc[kind]
    model.save(MODELS_DIR / "logistic_regression.json", extra={
        "encoding": kind, "input_names": enc_obj.output_names, "threshold": threshold,
        "threshold_rule": "maximise validation F1", "selected_by": "minimum validation log-loss",
        "trained_on": "train partition (4,004 rows)"})
    out = {
        "selected": {"encoding": kind, "l2": l2, "learning_rate": lr, "iterations": model.n_iter_run,
                     "converged": model.converged, "threshold": threshold},
        "val_metrics_at_threshold": compute_metrics(yva, pv, threshold),
        "train_metrics_at_threshold": compute_metrics(ytr, model.predict_proba(Xtr), threshold),
        "smoothness_constant_L": smooth, "stability_bound_2_over_L": two_over_L,
        "fit_seconds": fit_s, "peak_fit_memory_mb": peak_mb, "inference_ms_per_858_rows": infer_ms,
        "grid": grid, "learning_rate_study": lr_summary, "variants_val": variants,
        "reference_check": reference, "figures": figs,
    }
    update_results("logistic_regression", out)
    return out


# ------------------------------------------------------------- stage: decision tree

TREE_DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
TREE_LEAVES = [1, 2, 5, 10, 20]


def stage_tree() -> dict[str, Any]:
    """Phase 5: single custom tree — depth / leaf-size study on VALIDATION, sklearn check."""
    plotting.style()
    data, enc = get_data()
    m = matrices(data, enc["raw"])
    (Xtr, ytr), (Xva, yva) = m["train"], m["val"]
    grid = []
    for d in TREE_DEPTHS:
        for leaf in TREE_LEAVES:
            t0 = time.perf_counter()
            t = DecisionTree(max_depth=d, min_samples_leaf=leaf).fit(Xtr, ytr)
            secs = time.perf_counter() - t0
            mt, mv = compute_metrics(ytr, t.predict_proba(Xtr)), compute_metrics(yva, t.predict_proba(Xva))
            grid.append({"max_depth": d, "min_samples_leaf": leaf, "fit_seconds": secs, "n_leaves": t.n_leaves,
                         "depth": t.depth, "train_accuracy": mt["accuracy"], "train_roc_auc": mt["roc_auc"],
                         "train_log_loss": mt["log_loss"], "val_accuracy": mv["accuracy"],
                         "val_roc_auc": mv["roc_auc"], "val_log_loss": mv["log_loss"], "val_f1": mv["f1"]})
    best = min(grid, key=lambda g: (round(g["val_log_loss"], 4), g["n_leaves"]))
    params = {"max_depth": best["max_depth"], "min_samples_leaf": best["min_samples_leaf"]}
    tree = DecisionTree(**params).fit(Xtr, ytr)
    pv = tree.predict_proba(Xva)
    thr = tune_threshold(yva, pv, "f1")

    sk = sklearn_tree(params).fit(Xtr, ytr)
    psk = sk_proba(sk, Xva)
    unconstrained = next(g for g in grid if g["max_depth"] is None and g["min_samples_leaf"] == 1)

    d_lbl = [str(d) if d is not None else "∞" for d in TREE_DEPTHS]
    xs = list(range(len(TREE_DEPTHS)))
    leaf1 = [g for g in grid if g["min_samples_leaf"] == 1]
    leaf_best = [g for g in grid if g["min_samples_leaf"] == params["min_samples_leaf"]]
    figs = {"depth_curve": plotting.line_panel(
        {"train (leaf ≥ 1)": (xs, [g["train_log_loss"] for g in leaf1], "--", plotting.SERIES[0]),
         "validation (leaf ≥ 1)": (xs, [g["val_log_loss"] for g in leaf1], "-", plotting.SERIES[0]),
         f"train (leaf ≥ {params['min_samples_leaf']})": (xs, [g["train_log_loss"] for g in leaf_best], "--",
                                                          plotting.SERIES[1]),
         f"validation (leaf ≥ {params['min_samples_leaf']})": (xs, [g["val_log_loss"] for g in leaf_best], "-",
                                                               plotting.SERIES[1])},
        "max_depth (∞ = unlimited)", "Log-loss (clipped)", "Decision tree: depth vs train/validation loss",
        "tree_01_depth_curve.png", xticklabels=d_lbl)}
    out = {
        "selected": {**params, "n_leaves": tree.n_leaves, "depth": tree.depth, "threshold": thr},
        "val_metrics_at_threshold": compute_metrics(yva, pv, thr),
        "train_metrics_at_threshold": compute_metrics(ytr, tree.predict_proba(Xtr), thr),
        "unconstrained_tree": unconstrained,
        "grid": grid,
        "depth_labels": d_lbl,
        "sklearn_reference": {
            "n_leaves": int(sk.get_n_leaves()), "depth": int(sk.get_depth()),
            "val_metrics_at_threshold": compute_metrics(yva, psk, thr),
            "val_label_agreement": float(np.mean((psk >= thr) == (pv >= thr))),
            "max_abs_val_proba_diff": float(np.max(np.abs(psk - pv))),
        },
        "top_of_tree": tree.export_text(data.features, max_depth=3),
        "figures": figs,
    }
    update_results("decision_tree", out)
    return out


# ------------------------------------------------------------- stage: random forest

RF_GRID = {"max_depth": [None, 12, 8], "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 10, 15]}
RF_TREES = 100


def stage_rf() -> dict[str, Any]:
    """Phase 6: tune the custom forest on VALIDATION, study trees / features / bootstrap, save it."""
    plotting.style()
    data, enc = get_data()
    m = matrices(data, enc["raw"])
    (Xtr, ytr), (Xva, yva) = m["train"], m["val"]

    grid = []
    for d in RF_GRID["max_depth"]:
        for leaf in RF_GRID["min_samples_leaf"]:
            for mf in RF_GRID["max_features"]:
                rf = RandomForest(n_estimators=RF_TREES, max_depth=d, min_samples_leaf=leaf, max_features=mf,
                                  seed=RANDOM_SEED).fit(Xtr, ytr)
                mt, mv = compute_metrics(ytr, rf.predict_proba(Xtr)), compute_metrics(yva, rf.predict_proba(Xva))
                grid.append({"max_depth": d, "min_samples_leaf": leaf, "max_features": mf,
                             "fit_seconds": rf.fit_seconds_, "train_log_loss": mt["log_loss"],
                             "train_accuracy": mt["accuracy"], "val_log_loss": mv["log_loss"],
                             "val_roc_auc": mv["roc_auc"], "val_f1_at_0.5": mv["f1"], "val_accuracy": mv["accuracy"]})
    best = min(grid, key=lambda g: round(g["val_log_loss"], 4))
    params = {k: best[k] for k in ("max_depth", "min_samples_leaf", "max_features")}

    # Final forest: trained on train only, with OOB tracking; memory and time measured.
    tracemalloc.start()
    forest = RandomForest(n_estimators=RF_TREES, oob_score=True, seed=RANDOM_SEED, **params).fit(Xtr, ytr)
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    pv = forest.predict_proba(Xva)
    thr = tune_threshold(yva, pv, "f1")
    t0 = time.perf_counter()
    for _ in range(5):
        forest.predict_proba(Xva)
    infer_ms = (time.perf_counter() - t0) / 5 * 1000

    # Number of trees: running averages of a 300-tree forest (same params).
    big = RandomForest(n_estimators=300, oob_score=False, seed=RANDOM_SEED, **params).fit(Xtr, ytr)
    st_va, st_tr = big.staged_proba(Xva), big.staged_proba(Xtr)
    T = [1, 2, 5, 10, 25, 50, 100, 150, 200, 300]
    trees_curve = {"n_trees": T,
                   "train_log_loss": [compute_metrics(ytr, st_tr[t - 1])["log_loss"] for t in T],
                   "val_log_loss": [compute_metrics(yva, st_va[t - 1])["log_loss"] for t in T],
                   "val_roc_auc": [compute_metrics(yva, st_va[t - 1])["roc_auc"] for t in T],
                   "val_f1_at_0.5": [compute_metrics(yva, st_va[t - 1])["f1"] for t in T]}

    # Candidate features per split (others fixed), and bootstrap vs none.
    mf_study = {}
    for mf in (1, 2, 3, 5, 10, 15, 20, 30):
        rf = RandomForest(n_estimators=RF_TREES, seed=RANDOM_SEED, **{**params, "max_features": mf}).fit(Xtr, ytr)
        per_tree = np.array([t.predict_proba(Xva) for t in rf.trees])
        corr = np.corrcoef(per_tree)
        mf_study[str(mf)] = {"val_log_loss": compute_metrics(yva, per_tree.mean(0))["log_loss"],
                             "val_roc_auc": compute_metrics(yva, per_tree.mean(0))["roc_auc"],
                             "mean_tree_val_roc_auc": float(np.mean([compute_metrics(yva, p)["roc_auc"]
                                                                     for p in per_tree])),
                             "mean_pairwise_tree_correlation": float(corr[np.triu_indices_from(corr, 1)].mean()),
                             "fit_seconds": rf.fit_seconds_}
    nob = RandomForest(n_estimators=RF_TREES, bootstrap=False, seed=RANDOM_SEED, **params).fit(Xtr, ytr)
    bootstrap_study = {"bootstrap": compute_metrics(yva, pv, 0.5),
                       "no_bootstrap": compute_metrics(yva, nob.predict_proba(Xva), 0.5)}

    # Independent scikit-learn REFERENCE forest with the same hyperparameters.
    t0 = time.perf_counter()
    skf = sklearn_forest({"n_estimators": RF_TREES, **params}).fit(Xtr, ytr)
    sk_fit = time.perf_counter() - t0
    psk = sk_proba(skf, Xva)
    reference = {"val_metrics_at_threshold": compute_metrics(yva, psk, thr), "fit_seconds": sk_fit,
                 "val_proba_correlation": float(np.corrcoef(psk, pv)[0, 1]),
                 "val_label_agreement_at_threshold": float(np.mean((psk >= thr) == (pv >= thr)))}

    path = MODELS_DIR / "random_forest.json"
    forest.save(path, extra={"encoding": "raw", "input_names": enc["raw"].output_names, "threshold": thr,
                             "threshold_rule": "maximise validation F1",
                             "selected_by": "minimum validation log-loss",
                             "trained_on": "train partition (4,004 rows)"})

    figs = {
        "n_trees": plotting.line_panel(
            {"train": (T, trees_curve["train_log_loss"], "--", plotting.SERIES[0]),
             "validation": (T, trees_curve["val_log_loss"], "-", plotting.SERIES[0])},
            "Number of trees (log scale)", "Log-loss (clipped)", "Random forest: effect of the number of trees",
            "rf_01_n_trees.png", logx=True, logy=True),
        "max_features": plotting.line_panel(
            {"forest validation ROC-AUC": (list(range(8)), [v["val_roc_auc"] for v in mf_study.values()]),
             "mean single-tree validation ROC-AUC": (list(range(8)),
                                                     [v["mean_tree_val_roc_auc"] for v in mf_study.values()]),
             "mean pairwise tree correlation": (list(range(8)),
                                                [v["mean_pairwise_tree_correlation"] for v in mf_study.values()])},
            "max_features (candidate features per split)", "Value",
            "Feature subsampling: diversity vs strength", "rf_02_max_features.png", xticklabels=list(mf_study)),
        "threshold": plotting.threshold_plot(threshold_curve(yva, pv), thr, "rf_03_threshold.png",
                                             "Custom RF: validation precision / recall / F1 vs threshold"),
    }
    ok = ~np.isnan(forest.oob_proba_)
    out = {
        "selected": {**params, "n_estimators": RF_TREES, "threshold": thr},
        "val_metrics_at_threshold": compute_metrics(yva, pv, thr),
        "train_metrics_at_threshold": compute_metrics(ytr, forest.predict_proba(Xtr), thr),
        "oob_metrics_at_threshold": compute_metrics(ytr[ok], forest.oob_proba_[ok], thr),
        "fit_seconds": forest.fit_seconds_, "peak_fit_memory_mb": peak_mb,
        "inference_ms_per_858_rows": infer_ms, "model_file_mb": path.stat().st_size / 2**20,
        "mean_leaves_per_tree": float(np.mean([t.n_leaves for t in forest.trees])),
        "mean_depth_per_tree": float(np.mean([t.depth for t in forest.trees])),
        "grid": grid, "n_trees_curve": trees_curve, "max_features_study": mf_study,
        "bootstrap_study": bootstrap_study, "reference_check": reference, "figures": figs,
    }
    update_results("random_forest", out)
    return out


# ------------------------------------------------------------- stage: evaluation


def _selected() -> dict[str, dict]:
    res = load_results()
    missing = [k for k in ("logistic_regression", "random_forest", "decision_tree") if k not in res]
    if missing:
        raise RuntimeError(f"run the earlier stages first: {missing}")
    return {"lr": res["logistic_regression"]["selected"], "rf": res["random_forest"]["selected"],
            "tree": res["decision_tree"]["selected"], "best_feature": res["baselines_val"]["best_single_feature"]["feature"]}


def _per_class(m: dict[str, float]) -> dict[str, float]:
    """Precision/recall/F1 for both classes from confusion counts."""
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    (pp, rp, fp1), (pl, rl, fl) = prf(tp, fp, fn), prf(tn, fn, fp)
    return {"phishing_precision": pp, "phishing_recall": rp, "phishing_f1": fp1,
            "legitimate_precision": pl, "legitimate_recall": rl, "legitimate_f1": fl}


def stage_evaluate() -> dict[str, Any]:
    """Phase 7: 5-fold CV on train, then ONE evaluation of the frozen models on the test set."""
    plotting.style()
    sel = _selected()
    data, enc = get_data()
    raw, oh = matrices(data, enc["raw"]), matrices(data, enc["onehot"])
    lr_enc = raw if sel["lr"]["encoding"] == "raw" else oh
    Xtr, ytr = raw["train"]
    lr_params = {"learning_rate": sel["lr"]["learning_rate"], "l2": sel["lr"]["l2"], "n_iter": LR_MAX_ITER, "tol": LR_TOL}
    rf_params = {k: sel["rf"][k] for k in ("max_depth", "min_samples_leaf", "max_features", "n_estimators")}
    tree_params = {k: sel["tree"][k] for k in ("max_depth", "min_samples_leaf")}
    j_best = data.features.index(sel["best_feature"])

    # ---- 1. Stratified 5-fold CV on the training partition (thresholds fixed from validation).
    folds = stratified_folds(ytr, 5)
    cv_models = {
        "Custom LR": (lambda a, b, c: LogisticRegression(**lr_params).fit(a, b).predict_proba(c), lr_enc, sel["lr"]["threshold"]),
        "Custom RF": (lambda a, b, c: RandomForest(seed=RANDOM_SEED, **rf_params).fit(a, b).predict_proba(c), raw, sel["rf"]["threshold"]),
        "Custom tree": (lambda a, b, c: DecisionTree(**tree_params).fit(a, b).predict_proba(c), raw, sel["tree"]["threshold"]),
        "sklearn LR (ref)": (lambda a, b, c: sk_proba(sklearn_logistic(sel["lr"]["l2"], len(b)).fit(a, b), c), lr_enc, sel["lr"]["threshold"]),
        "sklearn RF (ref)": (lambda a, b, c: sk_proba(sklearn_forest(rf_params).fit(a, b), c), raw, sel["rf"]["threshold"]),
        "Best single feature": (lambda a, b, c: SingleFeatureBaseline(j_best).fit(a, b).predict_proba(c), raw, 0.5),
        "Majority": (lambda a, b, c: MajorityBaseline().fit(a, b).predict_proba(c), raw, 0.5),
    }
    cv_raw = {name: cross_validate(fn, mats["train"][0], ytr, folds, thr) for name, (fn, mats, thr) in cv_models.items()}
    cv = {
        "folds": 5, "seed": RANDOM_SEED, "partition": "train (4,004 rows)",
        "summary": {k: summarise(v) for k, v in cv_raw.items()},
        "per_fold": cv_raw,
        "paired_rf_minus_lr": {m: paired_fold_difference(cv_raw["Custom RF"], cv_raw["Custom LR"], m)
                               for m in ("roc_auc", "f1", "log_loss")},
        "paired_rf_minus_tree": {m: paired_fold_difference(cv_raw["Custom RF"], cv_raw["Custom tree"], m)
                                 for m in ("roc_auc", "f1", "log_loss")},
    }

    # ---- 2. Test set: frozen models (loaded from disk) and references trained on train only.
    lr_model = LogisticRegression.load(MODELS_DIR / "logistic_regression.json")
    rf_model = RandomForest.load(MODELS_DIR / "random_forest.json")
    Xte_raw, yte = raw["test"]
    Xte_lr = lr_enc["test"][0]
    tree = DecisionTree(**tree_params).fit(Xtr, ytr)
    probas = {
        "Custom LR": (lr_model.predict_proba(Xte_lr), sel["lr"]["threshold"]),
        "Custom RF": (rf_model.predict_proba(Xte_raw), sel["rf"]["threshold"]),
        "Custom tree": (tree.predict_proba(Xte_raw), sel["tree"]["threshold"]),
        "sklearn LR (ref)": (sk_proba(sklearn_logistic(sel["lr"]["l2"], len(ytr)).fit(lr_enc["train"][0], ytr), Xte_lr),
                             sel["lr"]["threshold"]),
        "sklearn RF (ref)": (sk_proba(sklearn_forest(rf_params).fit(Xtr, ytr), Xte_raw), sel["rf"]["threshold"]),
        "Best single feature": (SingleFeatureBaseline(j_best).fit(Xtr, ytr).predict_proba(Xte_raw), 0.5),
        "Majority": (MajorityBaseline().fit(Xtr, ytr).predict_proba(Xte_raw), 0.5),
    }
    test = {name: {**compute_metrics(yte, p, t), **_per_class(compute_metrics(yte, p, t))} for name, (p, t) in probas.items()}
    test_at_05 = {name: compute_metrics(yte, p, 0.5) for name, (p, _) in probas.items()}
    ci = {name: bootstrap_ci(yte, p, t, n_boot=1000) for name, (p, t) in probas.items() if name != "Majority"}

    def pair(a: str, b: str) -> dict[str, Any]:
        d = paired_bootstrap_diff(yte, probas[a][0], probas[a][1], probas[b][0], probas[b][1], n_boot=1000)
        return {m: {**v, "verdict": verdict(v)} for m, v in d.items()}

    comparisons = {
        "Custom RF vs Custom LR": pair("Custom RF", "Custom LR"),
        "Custom RF vs Custom tree": pair("Custom RF", "Custom tree"),
        "Custom LR vs Best single feature": pair("Custom LR", "Best single feature"),
        "Custom RF vs sklearn RF (ref)": pair("Custom RF", "sklearn RF (ref)"),
        "Custom LR vs sklearn LR (ref)": pair("Custom LR", "sklearn LR (ref)"),
    }

    # ---- 3. Duplicate diagnostic: identical pipeline on the RAW (non-deduplicated) data.
    from src.preprocessing import map_labels, split_indices
    rawdf = map_labels(load_raw())
    rs = split_indices(rawdf[TARGET])
    enc_raw_lr = FeatureEncoder(sel["lr"]["encoding"]).fit(rawdf.iloc[rs.train])
    Xr = enc["raw"].transform(rawdf)
    Xr_lr = enc_raw_lr.transform(rawdf)
    yr = rawdf[TARGET].to_numpy()
    diag_rf = RandomForest(seed=RANDOM_SEED, **rf_params).fit(Xr[rs.train], yr[rs.train]).predict_proba(Xr[rs.val])
    diag_lr = LogisticRegression(**lr_params).fit(Xr_lr[rs.train], yr[rs.train]).predict_proba(Xr_lr[rs.val])
    duplicate_diag = {
        "note": "Same hyperparameters, trained/evaluated on a split of the RAW 11,055 rows (duplicates kept). "
                "Diagnostic only — shows how duplicates inflate scores; never used for selection.",
        "raw_val_rows": int(len(rs.val)),
        "Custom RF raw-split val": compute_metrics(yr[rs.val], diag_rf, sel["rf"]["threshold"]),
        "Custom LR raw-split val": compute_metrics(yr[rs.val], diag_lr, sel["lr"]["threshold"]),
        "Custom RF clean val": load_results()["random_forest"]["val_metrics_at_threshold"],
        "Custom LR clean val": load_results()["logistic_regression"]["val_metrics_at_threshold"],
    }

    # ---- 4. Figures.
    main = ["Custom LR", "Custom RF", "sklearn LR (ref)", "sklearn RF (ref)", "Best single feature"]
    figs = {
        "roc_pr": plotting.roc_pr_curves(yte, {k: probas[k][0] for k in main}, "eval_01_roc_pr_test.png", "test set"),
        "confusion": plotting.confusion_grid({k: test[k] for k in ["Custom LR", "Custom RF", "Custom tree", "Best single feature"]},
                                             "eval_02_confusion_test.png", "Confusion matrices — test set (rows: actual)"),
    }
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    show = ["Custom LR", "Custom RF", "Custom tree", "sklearn RF (ref)", "Best single feature"]
    figs["comparison"] = plotting.grouped_bars(
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        {k: [test[k][m] for m in metrics] for k in show}, "eval_03_model_comparison_test.png",
        "Test-set metrics with 95% bootstrap CIs (positive = phishing)", "Score",
        errors={k: [(ci[k][m]["low"], ci[k][m]["high"]) for m in metrics] for k in show}, ylim=(0.8, 1.005))
    pc = ["phishing_precision", "phishing_recall", "phishing_f1", "legitimate_precision", "legitimate_recall", "legitimate_f1"]
    figs["per_class"] = plotting.grouped_bars(
        ["Phish P", "Phish R", "Phish F1", "Legit P", "Legit R", "Legit F1"],
        {k: [test[k][m] for m in pc] for k in ["Custom LR", "Custom RF", "Custom tree"]},
        "eval_04_per_class_test.png", "Per-class precision / recall / F1 — test set", "Score", ylim=(0.8, 1.005))
    figs["cv"] = plotting.grouped_bars(
        ["ROC-AUC", "PR-AUC", "F1", "Accuracy"],
        {k: [cv["summary"][k][m]["mean"] for m in ("roc_auc", "pr_auc", "f1", "accuracy")]
         for k in ["Custom LR", "Custom RF", "Custom tree", "sklearn LR (ref)", "sklearn RF (ref)"]},
        "eval_05_cv_train.png", "5-fold CV on the training partition (mean ± 1 sd)", "Score",
        errors={k: [(cv["summary"][k][m]["mean"] - cv["summary"][k][m]["std"],
                     cv["summary"][k][m]["mean"] + cv["summary"][k][m]["std"]) for m in ("roc_auc", "pr_auc", "f1", "accuracy")]
                for k in ["Custom LR", "Custom RF", "Custom tree", "sklearn LR (ref)", "sklearn RF (ref)"]},
        ylim=(0.85, 1.005))
    figs["threshold_test"] = plotting.threshold_plot(threshold_curve(yte, probas["Custom RF"][0]), sel["rf"]["threshold"],
                                                     "eval_06_rf_threshold_test.png",
                                                     "Custom RF: test precision / recall / F1 vs threshold (descriptive)")

    out = {"positive_class": "phishing (1)", "test_rows": int(len(yte)),
           "thresholds": {k: t for k, (_, t) in probas.items()},
           "cross_validation": cv, "test": test, "test_at_threshold_0.5": test_at_05, "test_bootstrap_ci": ci,
           "paired_comparisons_test": comparisons, "duplicate_diagnostic": duplicate_diag, "figures": figs}
    update_results("evaluation", out)
    return out


# ------------------------------------------------------- stage: overfitting + feature selection


def _subset_encoders(data: PreparedData, features: list[str]) -> dict[str, FeatureEncoder]:
    """Encoders restricted to ``features`` (fitted on train)."""
    train = data.part("train")[features + [TARGET]]
    return {k: FeatureEncoder(k).fit(train) for k in ("raw", "onehot")}


def stage_overfit() -> dict[str, Any]:
    """Phase 8: learning curves, RF depth/leaf study, and feature selection (VALIDATION only)."""
    plotting.style()
    sel = _selected()
    data, enc = get_data()
    raw = matrices(data, enc["raw"])
    lr_kind = sel["lr"]["encoding"]
    lr_m = matrices(data, enc[lr_kind])
    (Xtr, ytr), (Xva, yva) = raw["train"], raw["val"]
    lr_params = {"learning_rate": sel["lr"]["learning_rate"], "l2": sel["lr"]["l2"], "n_iter": LR_MAX_ITER, "tol": LR_TOL}
    rf_params = {k: sel["rf"][k] for k in ("max_depth", "min_samples_leaf", "max_features", "n_estimators")}
    tree_params = {k: sel["tree"][k] for k in ("max_depth", "min_samples_leaf")}
    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Learning curves over training-set size (stratified subsamples of train).
    fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    lc: dict[str, dict[str, list]] = {k: {"n": [], "train_log_loss": [], "val_log_loss": [], "train_roc_auc": [],
                                          "val_roc_auc": []} for k in ("Custom LR", "Custom RF", "Custom tree")}
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    for fr in fracs:
        sub = np.sort(np.r_[rng.permutation(pos)[: int(fr * len(pos))], rng.permutation(neg)[: int(fr * len(neg))]])
        fits = {
            "Custom LR": (LogisticRegression(**lr_params).fit(lr_m["train"][0][sub], ytr[sub]), lr_m),
            "Custom RF": (RandomForest(seed=RANDOM_SEED, **rf_params).fit(Xtr[sub], ytr[sub]), raw),
            "Custom tree": (DecisionTree(**tree_params).fit(Xtr[sub], ytr[sub]), raw),
        }
        for k, (model, mats) in fits.items():
            mt = compute_metrics(ytr[sub], model.predict_proba(mats["train"][0][sub]))
            mv = compute_metrics(yva, model.predict_proba(mats["val"][0]))
            lc[k]["n"].append(int(len(sub)))
            for part, mm in (("train", mt), ("val", mv)):
                lc[k][f"{part}_log_loss"].append(mm["log_loss"])
                lc[k][f"{part}_roc_auc"].append(mm["roc_auc"])
    figs = {"learning_curves": plotting.line_panel(
        {f"{k} – {part}": (v["n"], v[f"{part}_roc_auc"], "--" if part == "train" else "-", plotting.SERIES[i])
         for i, (k, v) in enumerate(lc.items()) for part in ("train", "val")},
        "Training rows used", "ROC-AUC", "Learning curves: train vs validation ROC-AUC", "ovf_01_learning_curves.png")}

    # 2. Random-forest depth study (other selected settings fixed) and tree leaf-size study.
    rf_depth = {}
    for d in (2, 4, 6, 8, 12, 15, None):
        rf = RandomForest(seed=RANDOM_SEED, **{**rf_params, "max_depth": d}).fit(Xtr, ytr)
        rf_depth[str(d)] = {"train_log_loss": compute_metrics(ytr, rf.predict_proba(Xtr))["log_loss"],
                            "val_log_loss": compute_metrics(yva, rf.predict_proba(Xva))["log_loss"],
                            "train_accuracy": compute_metrics(ytr, rf.predict_proba(Xtr))["accuracy"],
                            "val_accuracy": compute_metrics(yva, rf.predict_proba(Xva))["accuracy"]}
    tree_grid = load_results()["decision_tree"]["grid"]
    tree_leaf = {str(g["min_samples_leaf"]): {k: g[k] for k in ("train_log_loss", "val_log_loss", "n_leaves")}
                 for g in tree_grid if g["max_depth"] is None}
    xs = list(range(len(rf_depth)))
    unl = [g for g in tree_grid if g["min_samples_leaf"] == 1]
    tree_by_depth = {str(g["max_depth"]): g for g in unl}
    figs["rf_vs_tree_depth"] = plotting.line_panel(
        {"RF – train": (xs, [v["train_log_loss"] for v in rf_depth.values()], "--", plotting.SERIES[1]),
         "RF – validation": (xs, [v["val_log_loss"] for v in rf_depth.values()], "-", plotting.SERIES[1]),
         "single tree – train": (xs, [tree_by_depth[k]["train_log_loss"] for k in rf_depth], "--", plotting.SERIES[0]),
         "single tree – validation": (xs, [tree_by_depth[k]["val_log_loss"] for k in rf_depth], "-", plotting.SERIES[0])},
        "max_depth (None = unlimited)", "Log-loss (clipped, log scale)",
        "Depth: single tree overfits, forest does not", "ovf_02_depth_tree_vs_rf.png",
        logy=True, xticklabels=list(rf_depth))

    # 3. Feature selection (train-only), evaluated on validation.
    feats = data.features
    share = dominant_level_share(data.part("train"), feats)
    mi = mutual_information_ranking(data.part("train"), feats, TARGET)

    def evaluate_subset(fs: list[str]) -> dict[str, Any]:
        e = _subset_encoders(data, fs)
        r, o = matrices(data, e["raw"]), matrices(data, e[lr_kind])
        t0 = time.perf_counter()
        lr = LogisticRegression(**lr_params).fit(o["train"][0], ytr)
        lr_s = time.perf_counter() - t0
        rf = RandomForest(seed=RANDOM_SEED, **{**rf_params, "max_features": min(rf_params["max_features"], len(fs))
                                               if isinstance(rf_params["max_features"], int) else rf_params["max_features"]}
                          ).fit(r["train"][0], ytr)
        plr, prf = lr.predict_proba(o["val"][0]), rf.predict_proba(r["val"][0])
        return {"n_features": len(fs), "features": fs,
                "lr_val": compute_metrics(yva, plr, tune_threshold(yva, plr)), "lr_fit_seconds": lr_s,
                "rf_val": compute_metrics(yva, prf, tune_threshold(yva, prf)), "rf_fit_seconds": rf.fit_seconds_}

    fs_results: dict[str, Any] = {"all_features": evaluate_subset(feats)}
    for tau in (0.95, 0.90, 0.85):
        kept = near_constant_filter(data.part("train"), feats, tau)
        fs_results[f"near_constant_tau_{tau}"] = evaluate_subset(kept)
    for k in (3, 5, 8, 10, 15, 20):
        fs_results[f"mi_top_{k}"] = evaluate_subset(list(mi.index[:k]))
    html_cluster = ["favicon", "port", "popupwindow", "on_mouseover", "iframe", "submitting_to_email", "rightclick"]
    ablations = {"drop_prefix_suffix": ["prefix_suffix"], "drop_sslfinal_state": ["sslfinal_state"],
                 "drop_top2_sslfinal_url_of_anchor": ["sslfinal_state", "url_of_anchor"],
                 "drop_html_js_cluster": html_cluster,
                 "drop_domain_registration_length": ["domain_registration_length"]}
    for name, drop in ablations.items():
        fs_results[f"ablation_{name}"] = evaluate_subset([f for f in feats if f not in drop])

    ks = [3, 5, 8, 10, 15, 20, 30]
    rows = [fs_results[f"mi_top_{k}"] for k in ks[:-1]] + [fs_results["all_features"]]
    figs["mi_top_k"] = plotting.line_panel(
        {"Custom LR": (ks, [r["lr_val"]["roc_auc"] for r in rows]),
         "Custom RF": (ks, [r["rf_val"]["roc_auc"] for r in rows])},
        "Top-k features by training mutual information", "Validation ROC-AUC",
        "Feature selection: validation ROC-AUC vs number of features", "ovf_03_mi_top_k.png")
    figs["ablation"] = plotting.bar_h(
        {name.replace("ablation_", ""): fs_results[name]["rf_val"]["roc_auc"] - fs_results["all_features"]["rf_val"]["roc_auc"]
         for name in fs_results if name.startswith("ablation_") or name.startswith("near_constant")},
        "Change in RF validation ROC-AUC vs all 30 features", "Ablations and filters (Random Forest, validation)",
        "ovf_04_ablation_rf.png", signed=True)

    out = {"learning_curves": lc, "rf_depth_study": rf_depth, "tree_min_leaf_study_unlimited_depth": tree_leaf,
           "dominant_level_share_train": share.round(4).to_dict(),
           "raw_variance_train": data.part("train")[feats].var().round(4).to_dict(),
           "mutual_information_bits_train": mi.round(4).to_dict(),
           "feature_selection_val": fs_results, "figures": figs}
    update_results("overfitting_feature_selection", out)
    return out


# ------------------------------------------------------------- stage: interpretation


def stage_interpret() -> dict[str, Any]:
    """Phase 9: coefficients, impurity and permutation importance, stability across seeds/folds."""
    plotting.style()
    sel = _selected()
    data, enc = get_data()
    lr_kind = sel["lr"]["encoding"]
    lr_enc = enc[lr_kind]
    raw, lrm = matrices(data, enc["raw"]), matrices(data, lr_enc)
    ytr, yva = data.y("train"), data.y("val")
    lr_model = LogisticRegression.load(MODELS_DIR / "logistic_regression.json")
    rf_model = RandomForest.load(MODELS_DIR / "random_forest.json")
    feats = data.features
    rf_params = {k: sel["rf"][k] for k in ("max_depth", "min_samples_leaf", "max_features", "n_estimators")}

    # 1. LR coefficients (log-odds change vs the reference level of each feature).
    coefs = dict(zip(lr_enc.output_names, lr_model.w.tolist()))
    figs = {"lr_coefficients": plotting.bar_h(
        coefs, "Coefficient (log-odds of phishing vs reference level)",
        f"Custom LR coefficients ({lr_kind}; reference = level 1 of each feature)", "int_01_lr_coefficients.png",
        signed=True)}

    # Regularisation path: how coefficients shrink as λ grows.
    path_l2 = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
    top_names = [n for n, _ in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:6]]
    path = {}
    for lam in path_l2:
        lr = 0.9 / LogisticRegression.smoothness_constant(lrm["train"][0], lam)
        mdl = LogisticRegression(learning_rate=lr, l2=lam, n_iter=LR_MAX_ITER, tol=LR_TOL).fit(lrm["train"][0], ytr)
        path[str(lam)] = {"weights": dict(zip(lr_enc.output_names, mdl.w.tolist())), "l2_norm": float(np.linalg.norm(mdl.w)),
                          "iterations": mdl.n_iter_run}
    xs = list(range(len(path_l2)))
    figs["l2_path"] = plotting.line_panel(
        {n: (xs, [path[str(l)]["weights"][n] for l in path_l2]) for n in top_names},
        "L2 strength λ", "Coefficient", "Regularisation path of the six largest coefficients",
        "int_02_l2_path.png", xticklabels=[f"{l:g}" for l in path_l2])

    # 2. Impurity importance (RF) and permutation importance on VALIDATION (both models).
    imp = dict(zip(feats, rf_model.feature_importances_.tolist()))
    val_frame = data.part("val")
    lr_pred = lambda df: lr_model.predict_proba(lr_enc.transform(df))  # noqa: E731
    rf_pred = lambda df: rf_model.predict_proba(enc["raw"].transform(df))  # noqa: E731
    perm_lr = permutation_importance(lr_pred, val_frame, yva, feats, n_repeats=10)
    perm_rf = permutation_importance(rf_pred, val_frame, yva, feats, n_repeats=10)
    figs["rf_impurity"] = plotting.bar_h(imp, "Mean decrease in Gini impurity (normalised)",
                                         "Custom RF impurity importance (training data)", "int_03_rf_impurity.png")
    figs["perm"] = plotting.grouped_bars(
        list(perm_rf["feature"][:10]),
        {"Custom RF": [float(perm_rf.set_index("feature").loc[f, "mean_auc_drop"]) for f in perm_rf["feature"][:10]],
         "Custom LR": [float(perm_lr.set_index("feature").loc[f, "mean_auc_drop"]) for f in perm_rf["feature"][:10]]},
        "int_04_permutation_importance.png", "Permutation importance on validation (top 10 for RF)",
        "Mean ROC-AUC drop (10 shuffles)")

    # 3. Stability: RF impurity importance across 5 seeds; LR coefficients across 5 CV folds.
    seed_imps = np.array([RandomForest(seed=s, **rf_params).fit(raw["train"][0], ytr).feature_importances_
                          for s in range(5)])
    lr_params = {"learning_rate": sel["lr"]["learning_rate"], "l2": sel["lr"]["l2"], "n_iter": LR_MAX_ITER, "tol": LR_TOL}
    fold_coefs = np.array([LogisticRegression(**lr_params).fit(lrm["train"][0][a], ytr[a]).w
                           for a, _ in stratified_folds(ytr, 5)])
    stability = {
        "rf_impurity_across_seeds": {**rank_stability(seed_imps, feats),
                                     "std_by_feature": dict(zip(feats, seed_imps.std(0).round(5).tolist()))},
        "lr_coefficients_across_folds": {**rank_stability(np.abs(fold_coefs), lr_enc.output_names),
                                         "mean": dict(zip(lr_enc.output_names, fold_coefs.mean(0).round(4).tolist())),
                                         "std": dict(zip(lr_enc.output_names, fold_coefs.std(0, ddof=1).round(4).tolist())),
                                         "sign_flips": [n for n, col in zip(lr_enc.output_names, fold_coefs.T)
                                                        if (col > 0).any() and (col < 0).any()]},
    }
    # Rank agreement between the importance views.
    rank = lambda s: s.rank(ascending=False)  # noqa: E731
    views = pd.DataFrame({"rf_impurity": pd.Series(imp),
                          "rf_permutation": perm_rf.set_index("feature")["mean_auc_drop"],
                          "lr_permutation": perm_lr.set_index("feature")["mean_auc_drop"]})
    agreement = views.apply(rank).corr(method="spearman").round(3).to_dict()

    out = {"lr_coefficients": coefs, "lr_intercept": lr_model.b, "lr_l2_path": path,
           "rf_impurity_importance": imp, "permutation_importance_val": {
               "Custom LR": perm_lr.to_dict(orient="records"), "Custom RF": perm_rf.to_dict(orient="records")},
           "importance_rank_agreement_spearman": agreement, "stability": stability, "figures": figs}
    update_results("interpretation", out)
    return out


STAGES = {"prepare": stage_prepare, "baselines": stage_baselines, "lr": stage_lr, "tree": stage_tree,
          "rf": stage_rf, "evaluate": stage_evaluate, "overfit": stage_overfit, "interpret": stage_interpret}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), action="append")
    args = ap.parse_args(argv)
    for name in args.stage or list(STAGES):
        t0 = time.perf_counter()
        print(f"\n=== stage: {name} ===")
        res = STAGES[name]()
        _print_summary(name, res)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s")


def _print_summary(name: str, res: Any) -> None:
    if name == "baselines":
        cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        rows = {k: v for k, v in res.items() if isinstance(v, dict)}
        print(pd.DataFrame(rows).T[cols].astype(float).round(4).to_string())
        print("best single feature:", res["best_single_feature"]["feature"])
    elif name == "lr":
        print("selected:", res["selected"])
        print("val @thr:", {k: round(v, 4) for k, v in res["val_metrics_at_threshold"].items()})
        print("train @thr acc/f1:", round(res["train_metrics_at_threshold"]["accuracy"], 4),
              round(res["train_metrics_at_threshold"]["f1"], 4))
        g = pd.DataFrame(res["grid"])
        print(g.pivot_table(index=["encoding", "l2"], columns="lr_times_L", values="val_log_loss").round(4))
        print(g.pivot_table(index=["encoding", "l2"], columns="lr_times_L", values="iterations"))
        print("lr study:", {k: (round(v.get("final_train_objective", float("nan")), 5), v.get("monotone"))
                            for k, v in res["learning_rate_study"].items()})
        print("variants:", {k: (round(v["f1"], 4), round(v["recall"], 4), round(v["roc_auc"], 4))
                            for k, v in res["variants_val"].items()})
        print("reference check:", {k: v for k, v in res["reference_check"].items() if k != "sklearn_val_metrics"})
        print("fit s / peak MB / infer ms:", res["fit_seconds"], res["peak_fit_memory_mb"],
              res["inference_ms_per_858_rows"])
    elif name == "tree":
        print("selected:", res["selected"])
        print("val @thr:", {k: round(v, 4) for k, v in res["val_metrics_at_threshold"].items()})
        print("train @thr acc:", round(res["train_metrics_at_threshold"]["accuracy"], 4))
        u = res["unconstrained_tree"]
        print("unconstrained: leaves", u["n_leaves"], "depth", u["depth"], "train acc", round(u["train_accuracy"], 4),
              "val acc", round(u["val_accuracy"], 4), "val auc", round(u["val_roc_auc"], 4))
        g = pd.DataFrame(res["grid"])
        g["max_depth"] = g["max_depth"].fillna(-1).astype(int)
        print(g.pivot_table(index="max_depth", columns="min_samples_leaf", values="val_log_loss").round(4))
        print("sklearn ref:", {k: v for k, v in res["sklearn_reference"].items() if k != "val_metrics_at_threshold"})
        print(res["top_of_tree"])
    elif name == "rf":
        print("selected:", res["selected"])
        for k in ("val_metrics_at_threshold", "train_metrics_at_threshold", "oob_metrics_at_threshold"):
            print(k, {m: round(res[k][m], 4) for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")})
        g = pd.DataFrame(res["grid"]); g["max_depth"] = g["max_depth"].fillna(-1).astype(int)
        g["max_features"] = g["max_features"].astype(str)
        print(g.pivot_table(index=["max_depth", "min_samples_leaf"], columns="max_features", values="val_log_loss").round(4))
        print("n_trees:", dict(zip(res["n_trees_curve"]["n_trees"], [round(v, 4) for v in res["n_trees_curve"]["val_log_loss"]])))
        print("max_features:", {k: (round(v["val_roc_auc"], 4), round(v["mean_tree_val_roc_auc"], 4),
                                    round(v["mean_pairwise_tree_correlation"], 3)) for k, v in res["max_features_study"].items()})
        print("bootstrap vs none (val auc/f1):", {k: (round(v["roc_auc"], 4), round(v["f1"], 4)) for k, v in res["bootstrap_study"].items()})
        r = res["reference_check"]
        print("sklearn ref:", {m: round(r["val_metrics_at_threshold"][m], 4) for m in ("f1", "roc_auc", "log_loss")},
              "corr", round(r["val_proba_correlation"], 4), "agree", round(r["val_label_agreement_at_threshold"], 4),
              "fit s", round(r["fit_seconds"], 2))
        print("fit s / peak MB / infer ms / file MB / leaves / depth:", round(res["fit_seconds"], 2),
              round(res["peak_fit_memory_mb"], 1), round(res["inference_ms_per_858_rows"], 1),
              round(res["model_file_mb"], 2), res["mean_leaves_per_tree"], res["mean_depth_per_tree"])
    elif name == "evaluate":
        cols = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
        print("TEST (each model at its validation-chosen threshold):")
        print(pd.DataFrame(res["test"]).T[cols].astype(float).round(4).to_string())
        print("\nCV on train (mean +- sd):")
        print(pd.DataFrame({k: {m: f"{v[m]['mean']:.4f}+-{v[m]['std']:.4f}" for m in ("roc_auc", "f1", "log_loss")}
                            for k, v in res["cross_validation"]["summary"].items()}).T.to_string())
        print("CV paired RF-LR:", res["cross_validation"]["paired_rf_minus_lr"])
        print("\nTest 95% CI (F1, ROC-AUC):")
        for k, v in res["test_bootstrap_ci"].items():
            print(f"  {k:<22} F1 {v['f1']['point']:.4f} [{v['f1']['low']:.4f}, {v['f1']['high']:.4f}]  "
                  f"AUC {v['roc_auc']['point']:.4f} [{v['roc_auc']['low']:.4f}, {v['roc_auc']['high']:.4f}]")
        print("\nPaired test comparisons:")
        for k, v in res["paired_comparisons_test"].items():
            print(f"  {k}: " + "; ".join(f"{m} {d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] {d['verdict']}"
                                         for m, d in v.items() if m in ("f1", "roc_auc")))
        dd = res["duplicate_diagnostic"]
        print("\nDuplicate diagnostic (acc / f1 / auc):", {k: (round(v["accuracy"], 4), round(v["f1"], 4), round(v["roc_auc"], 4))
                                                         for k, v in dd.items() if isinstance(v, dict)})
    elif name == "overfit":
        for k, v in res["learning_curves"].items():
            print(k, "n:", v["n"], "train auc:", [round(x, 4) for x in v["train_roc_auc"]],
                  "val auc:", [round(x, 4) for x in v["val_roc_auc"]])
        print("RF depth:", {k: (round(v["train_log_loss"], 4), round(v["val_log_loss"], 4)) for k, v in res["rf_depth_study"].items()})
        print("MI top:", list(res["mutual_information_bits_train"].items())[:6])
        for k, v in res["feature_selection_val"].items():
            print(f"  {k:<44} n={v['n_features']:>2}  LR auc {v['lr_val']['roc_auc']:.4f} f1 {v['lr_val']['f1']:.4f} "
                  f"({v['lr_fit_seconds']:.1f}s)  RF auc {v['rf_val']['roc_auc']:.4f} f1 {v['rf_val']['f1']:.4f} ({v['rf_fit_seconds']:.1f}s)")
    elif name == "interpret":
        c = sorted(res["lr_coefficients"].items(), key=lambda kv: -abs(kv[1]))
        print("LR top coefs:", [(k, round(v, 3)) for k, v in c[:12]], "intercept", round(res["lr_intercept"], 3))
        print("RF impurity top:", sorted(((k, round(v, 4)) for k, v in res["rf_impurity_importance"].items()), key=lambda kv: -kv[1])[:10])
        for k, v in res["permutation_importance_val"].items():
            print(k, "perm top:", [(r["feature"], round(r["mean_auc_drop"], 4), round(r["std"], 4)) for r in v[:8]])
        print("rank agreement:", res["importance_rank_agreement_spearman"])
        st = res["stability"]
        print("RF seed stability:", round(st["rf_impurity_across_seeds"]["mean_pairwise_spearman"], 3),
              "min", round(st["rf_impurity_across_seeds"]["min_pairwise_spearman"], 3))
        print("LR fold stability:", round(st["lr_coefficients_across_folds"]["mean_pairwise_spearman"], 3),
              "sign flips:", st["lr_coefficients_across_folds"]["sign_flips"])
        print("L2 norms:", {k: round(v["l2_norm"], 2) for k, v in res["lr_l2_path"].items()})
    elif isinstance(res, dict):
        print(json.dumps(_round(res), indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
