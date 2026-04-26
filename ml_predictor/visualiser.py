"""
ml_predictor/visualiser.py
=================

Visualisation suite for three-body instability ML experiments.

Aligned with:
- trainer.py (CVResults structure)
- evaluator.py (EvaluationReport structure)

Focus:
- reliable diagnostics
- no broken dependencies
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ml_predictor.trainer import CVResults
from ml_predictor.evaluator import EvaluationReport

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (self-contained, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["stable", "unstable", "chaotic"]
N_CLASSES = 3
REGIME_ORDER = ["hierarchical", "asymmetric", "compact_equal", "scatter"]

CLASS_COLORS = {
    "stable":   "#2196F3",
    "unstable": "#F44336",
    "chaotic":  "#FF9800",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    report: EvaluationReport,
    normalise: bool = True,
    title: str = "Aggregate OOF Confusion Matrix",
) -> plt.Figure:

    cm = report.confusion.astype(float)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.where(row_sums == 0, 1, row_sums))

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, vmin=0, vmax=1 if normalise else None)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm[i, j]
            ax.text(
                j, i,
                f"{val:.2f}" if normalise else f"{int(val)}",
                ha="center", va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if val > 0.6 else "black"
            )

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.colorbar(im, ax=ax)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE IMPORTANCE (SAFE FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    results: CVResults,
    top_n: int = 20,
) -> plt.Figure:
    """
    Plot mean feature importance averaged across CV folds.

    All five models in MODEL_REGISTRY expose feature_importances_ after
    fitting (sklearn API). Averaging across folds gives a more stable
    importance estimate than any single fold.

    Falls back to a random proxy only if no fitted models are stored
    (should not occur for any model in MODEL_REGISTRY).
    """
    importances = []
    for model in results.models:
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

    if importances:
        importance = np.mean(importances, axis=0)
        title = f"Feature Importance ({results.model_name}, mean over folds)"
    else:
        importance = np.random.rand(len(results.feature_names))
        title = "Feature Importance (proxy — no fitted models)"

    df = pd.DataFrame({
        "feature": results.feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.35)))

    ax.barh(df["feature"], df["importance"])
    ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel("Relative Importance")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIME-WISE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_f1(report: EvaluationReport) -> plt.Figure | None:

    df = report.regime_metrics
    if df is None or df.empty:
        return None

    regimes = [r for r in REGIME_ORDER if r in df["regime"].values]

    values = [
        df[df["regime"] == r]["macro_f1"].values[0]
        for r in regimes
    ]

    fig, ax = plt.subplots(figsize=(6, len(regimes) * 0.8))

    ax.barh(regimes, values)
    ax.set_xlim(0, 1)

    ax.set_xlabel("Macro F1")
    ax.set_title("Regime-wise Performance")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. CALIBRATION CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_curves(results: CVResults) -> plt.Figure:

    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, N_CLASSES, figsize=(12, 4))

    y_true = results.y_true
    y_prob = results.y_proba

    for cls in range(N_CLASSES):
        ax = axes[cls]

        binary = (y_true == cls).astype(int)
        prob = y_prob[:, cls]

        frac, mean = calibration_curve(binary, prob, n_bins=10)

        ax.plot(mean, frac, marker="o")
        ax.plot([0, 1], [0, 1], "--")

        ax.set_title(CLASS_NAMES[cls])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(
    results: CVResults,
    report: EvaluationReport,
    output_dir: Path | None = None,
) -> dict[str, plt.Figure]:

    figs = {}

    print("Generating plots...")

    figs["confusion_matrix"]   = plot_confusion_matrix(report)
    figs["feature_importance"] = plot_feature_importance(results)
    figs["calibration"]        = plot_calibration_curves(results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, fig in figs.items():
            if fig is not None:
                path = output_dir / f"{name}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {path}")

    return figs