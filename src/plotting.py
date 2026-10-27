"""Shared figure style and evaluation plots (separate from training code)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import FIGURES_DIR, REPORTS_DIR
from src.eda import GRID, INK, INK_2, LEGIT, PHISH, SURFACE, _style

# Categorical slots (validated reference palette, fixed order).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

__all__ = ["plt", "style", "save", "SERIES", "INK", "INK_2", "GRID", "PHISH", "LEGIT", "SURFACE"]


def style() -> None:
    _style()


def save(fig: plt.Figure, name: str) -> str:
    """Save under reports/figures and return the repo-relative path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPORTS_DIR.parent)).replace("\\", "/")


def roc_pr_curves(y: np.ndarray, probas: dict[str, np.ndarray], name: str, title_suffix: str) -> str:
    """ROC and precision-recall curves for several models on one evaluation set."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, (label, p) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, color=SERIES[i], lw=2, label=label)
        a2.plot(rec, prec, color=SERIES[i], lw=2, label=label)
    a1.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1)
    a2.axhline(np.mean(y), ls="--", color=INK_2, lw=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title=f"ROC curves — {title_suffix}")
    a2.set(xlabel="Recall (phishing)", ylabel="Precision (phishing)", title=f"Precision–recall — {title_suffix}",
           ylim=(0.45, 1.01))
    a1.legend(loc="lower right", fontsize=8)
    a2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def confusion_grid(cms: dict[str, dict], name: str, title: str) -> str:
    """Confusion-matrix heatmaps (rows = actual, cols = predicted) with counts and row %."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    axes = np.atleast_1d(axes)
    for ax, (label, m) in zip(axes, cms.items()):
        mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        rowpct = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(rowpct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}\n{100 * rowpct[i, j]:.1f}%", ha="center", va="center",
                        color="white" if rowpct[i, j] > 0.6 else INK, fontsize=9)
        ax.set_xticks([0, 1], ["Legit", "Phish"])
        ax.set_yticks([0, 1], ["Legit", "Phish"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        ax.set_title(f"{label}\n(threshold {m['threshold']:.2f})", fontsize=9)
    fig.suptitle(title, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, name)


def line_panel(series: dict[str, tuple], xlabel: str, ylabel: str, title: str, name: str,
               logx: bool = False, logy: bool = False, ref: float | None = None,
               xticklabels: list[str] | None = None) -> str:
    """Generic multi-series line chart. ``series`` maps label -> (x, y[, linestyle[, colour]])."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (label, s) in enumerate(series.items()):
        x, y = s[0], s[1]
        ls = s[2] if len(s) > 2 else "-"
        color = s[3] if len(s) > 3 else SERIES[i % len(SERIES)]
        ax.plot(x, y, color=color, lw=2, ls=ls, label=label,
                marker="o" if len(x) <= 12 else None, markersize=4)
    if ref is not None:
        ax.axhline(ref, color=INK_2, ls=":", lw=1)
    if logx:
        ax.set_xscale("symlog" if min(min(s[0]) for s in series.values()) <= 0 else "log",
                      **({"linthresh": 1e-5} if min(min(s[0]) for s in series.values()) <= 0 else {}))
    if logy:
        ax.set_yscale("log")
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)), xticklabels)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def threshold_plot(curve: dict, chosen: float, name: str, title: str) -> str:
    """Precision / recall / F1 versus decision threshold with the chosen threshold marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, k in enumerate(("precision", "recall", "f1")):
        ax.plot(curve["threshold"], curve[k], color=SERIES[i], lw=2, label=k.capitalize() if k != "f1" else "F1")
    ax.axvline(chosen, color=INK_2, ls="--", lw=1)
    ax.text(chosen, 0.52, f" chosen {chosen:.2f}", color=INK_2, fontsize=8)
    ax.set(xlabel="Threshold on phishing probability", ylabel="Score (validation)", title=title, ylim=(0.5, 1.01))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def bar_h(values: dict[str, float], xlabel: str, title: str, name: str, err: dict | None = None,
          color: str | None = None, signed: bool = False) -> str:
    """Sorted horizontal bar chart (optionally with error bars or signed colouring)."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    labels, vals = [k for k, _ in items], [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.24 * len(items) + 1)))
    colors = [(PHISH if v > 0 else LEGIT) for v in vals] if signed else (color or SERIES[0])
    xerr = None
    if err:
        xerr = np.array([[v - err[k][0] for k, v in items], [err[k][1] - v for k, v in items]])
    ax.barh(labels, vals, color=colors, height=0.65, xerr=xerr, ecolor=INK_2, error_kw={"lw": 1})
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    ax.set(xlabel=xlabel, title=title)
    fig.tight_layout()
    return save(fig, name)
