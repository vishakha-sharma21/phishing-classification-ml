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
from src.evaluation import compute_metrics
from src.logistic_regression import LogisticRegression
from src.preprocessing import TARGET, FeatureEncoder, PreparedData, prepare, save_split
from src.reference_models import MajorityBaseline, SingleFeatureBaseline, sk_proba, sklearn_logistic
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


STAGES = {"prepare": stage_prepare, "baselines": stage_baselines, "lr": stage_lr}


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
    elif isinstance(res, dict):
        print(json.dumps(_round(res), indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
