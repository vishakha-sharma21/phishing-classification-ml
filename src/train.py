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
from typing import Any

import numpy as np
import pandas as pd

from src.config import LABEL_MAPPING, MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR
from src.data_loading import load_raw
from src.evaluation import compute_metrics
from src.preprocessing import TARGET, FeatureEncoder, PreparedData, prepare, save_split
from src.reference_models import MajorityBaseline, SingleFeatureBaseline, sk_proba, sklearn_logistic

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


STAGES = {"prepare": stage_prepare, "baselines": stage_baselines}


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
    elif isinstance(res, dict):
        print(json.dumps(_round(res), indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
