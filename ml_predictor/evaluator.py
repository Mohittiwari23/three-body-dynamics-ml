"""
ml_predictor/evaluator.py
===============

Evaluation module aligned with:
- trainer.py (CVResults)
- features.py (labels + regimes)
- run_baseline.py (save_results_csv)

Focus:
- OOF-based evaluation
- regime-aware diagnostics
- CSV persistence for baseline + hypertuned results
- minimal terminal output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["stable", "unstable", "chaotic"]
N_CLASSES = 3

REGIME_ORDER = ["hierarchical", "asymmetric", "compact_equal", "scatter"]


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    """
    Full OOF evaluation result for one (model, feature_group) run.

    Carries model_name and feature_group explicitly so that a flat list
    of EvaluationReport objects can be serialised to CSV without the caller
    having to re-attach metadata at save time.

    fold_macro_f1_std is stored separately from macro_f1 because the OOF
    aggregate F1 (computed on all held-out predictions pooled) differs
    subtly from the mean of per-fold F1s — the pooled version is the more
    reliable estimate on imbalanced datasets and is what macro_f1 reflects.
    fold_macro_f1_std reflects variance across folds, which is a useful
    stability diagnostic independent of the point estimate.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    model_name:    str = ""
    feature_group: str = ""

    # ── Aggregate OOF metrics ─────────────────────────────────────────────────
    macro_f1:      float = 0.0
    balanced_acc:  float = 0.0
    mean_brier:    float = 0.0

    # ── Fold-level stability ──────────────────────────────────────────────────
    # Sourced from CVResults.fold_macro_f1 — not recomputed here.
    fold_macro_f1_mean: float = 0.0
    fold_macro_f1_std:  float = 0.0

    # ── Per-class diagnostics ─────────────────────────────────────────────────
    per_class_recall:    dict = field(default_factory=dict)
    per_class_precision: dict = field(default_factory=dict)
    per_class_f1:        dict = field(default_factory=dict)

    # ── Confusion matrix (N_CLASSES × N_CLASSES) ──────────────────────────────
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=int))

    # ── Regime breakdown (None if regime column absent) ───────────────────────
    regime_metrics: pd.DataFrame | None = None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    results,
    df:            pd.DataFrame | None = None,
    model_name:    str                 = "",
    feature_group: str                 = "",
    verbose:       bool                = True,
) -> EvaluationReport:
    """
    Compute OOF evaluation metrics from a CVResults object.

    Parameters
    ----------
    results       : CVResults from train_model_cv or train_lgbm_tuned_cv.
    df            : Full dataset DataFrame. Required for regime breakdown.
    model_name    : Model identifier — stored on report for CSV traceability.
                    If empty, falls back to results.model_name.
    feature_group : Feature group label (A–F) — stored on report for CSV.
    verbose       : Print compact one-block summary to stdout.

    Returns
    -------
    EvaluationReport with all metrics and identity fields populated.
    """
    y_true  = results.y_true
    y_pred  = results.y_pred
    y_proba = results.y_proba

    # ── Overall OOF metrics ───────────────────────────────────────────────────
    # Pooled OOF computation — more reliable than mean(per-fold F1) on
    # imbalanced datasets because fold sizes differ and minority classes
    # may be represented unevenly across folds.
    macro_f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm           = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_f1        = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    per_recall    = _per_class_recall(y_true, y_pred)
    per_precision = _per_class_precision(y_true, y_pred)

    # ── Brier score (mean across classes) ────────────────────────────────────
    # Per-class Brier measures calibration independently for each outcome.
    # Mean Brier penalises overconfident wrong predictions — important here
    # because a model that assigns P(chaotic)=0.9 to a stable system is
    # far more dangerous operationally than one that is merely uncertain.
    brier_scores = [
        brier_score_loss((y_true == cls).astype(float), y_proba[:, cls])
        for cls in range(N_CLASSES)
    ]
    mean_brier = float(np.mean(brier_scores))

    # ── Regime breakdown ──────────────────────────────────────────────────────
    regime_metrics = None
    if df is not None and "regime" in df.columns:
        regime_metrics = _regime_breakdown(y_true, y_pred, df["regime"].values)

    # ── Assemble report ───────────────────────────────────────────────────────
    report = EvaluationReport(
        model_name    = model_name or results.model_name,
        feature_group = feature_group,

        macro_f1      = float(macro_f1),
        balanced_acc  = float(balanced_acc),
        mean_brier    = mean_brier,

        fold_macro_f1_mean = float(np.mean(results.fold_macro_f1)),
        fold_macro_f1_std  = float(np.std(results.fold_macro_f1)),

        per_class_recall    = {CLASS_NAMES[i]: float(per_recall[i])    for i in range(N_CLASSES)},
        per_class_precision = {CLASS_NAMES[i]: float(per_precision[i]) for i in range(N_CLASSES)},
        per_class_f1        = {CLASS_NAMES[i]: float(per_f1[i])        for i in range(N_CLASSES)},

        confusion      = cm,
        regime_metrics = regime_metrics,
    )

    if verbose:
        _print_report(report)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# REGIME BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def _regime_breakdown(y_true, y_pred, regimes):

    regimes = np.array(regimes)

    unique_regimes = [r for r in REGIME_ORDER if r in regimes]
    for r in np.unique(regimes):
        if r not in unique_regimes:
            unique_regimes.append(r)

    rows = []

    for regime in unique_regimes:
        mask = regimes == regime
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        f1  = f1_score(yt, yp, average="macro", zero_division=0)
        ba  = balanced_accuracy_score(yt, yp)
        rc  = _per_class_recall(yt, yp)
        pf1 = f1_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)

        rows.append({
            "regime":          regime,
            "n_samples":       int(mask.sum()),
            "macro_f1":        f1,
            "balanced_acc":    ba,
            "f1_stable":       pf1[0],
            "f1_unstable":     pf1[1],
            "f1_chaotic":      pf1[2],
            "recall_stable":   rc[0],
            "recall_unstable": rc[1],
            "recall_chaotic":  rc[2],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _per_class_recall(y_true, y_pred):
    recalls = np.zeros(N_CLASSES)
    for cls in range(N_CLASSES):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recalls


def _per_class_precision(y_true, y_pred):
    precisions = np.zeros(N_CLASSES)
    for cls in range(N_CLASSES):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precisions


# ─────────────────────────────────────────────────────────────────────────────
# CSV PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(
    reports:    list[EvaluationReport],
    output_dir: Path | str = Path("results/baseline"),
) -> pd.DataFrame:
    """
    Serialise a list of EvaluationReport objects to all_results.csv.

    One row per (model, feature_group) combination. Includes both the
    pooled OOF aggregate metrics and the fold-level stability estimates.
    Per-class F1/recall are flattened to individual columns so the CSV
    is directly queryable without JSON parsing.

    Parameters
    ----------
    reports    : Flat list of EvaluationReport — one per training run.
    output_dir : Directory to write all_results.csv into.

    Returns
    -------
    DataFrame with all rows.
    """
    rows = []

    for r in reports:
        row = {
            "model":          r.model_name,
            "feature_group":  r.feature_group,

            # Primary ranking metrics
            "macro_f1":       round(r.macro_f1,      4),
            "macro_f1_std":   round(r.fold_macro_f1_std, 4),
            "balanced_acc":   round(r.balanced_acc,  4),
            "mean_brier":     round(r.mean_brier,    4),

            # Fold-level mean (cross-check against pooled OOF)
            "fold_macro_f1_mean": round(r.fold_macro_f1_mean, 4),

            # Per-class F1 — key diagnostic for imbalanced three-body outcomes
            "f1_stable":    round(r.per_class_f1.get("stable",   0.0), 4),
            "f1_unstable":  round(r.per_class_f1.get("unstable", 0.0), 4),
            "f1_chaotic":   round(r.per_class_f1.get("chaotic",  0.0), 4),

            # Per-class recall — false negative rate proxy per outcome
            "recall_stable":    round(r.per_class_recall.get("stable",   0.0), 4),
            "recall_unstable":  round(r.per_class_recall.get("unstable", 0.0), 4),
            "recall_chaotic":   round(r.per_class_recall.get("chaotic",  0.0), 4),

            # Per-class precision
            "precision_stable":    round(r.per_class_precision.get("stable",   0.0), 4),
            "precision_unstable":  round(r.per_class_precision.get("unstable", 0.0), 4),
            "precision_chaotic":   round(r.per_class_precision.get("chaotic",  0.0), 4),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_acc"], ascending=False
    ).reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "all_results.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def leakage_sanity_check(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Test whether simulation metadata columns (dt, n_steps) predict outcome.

    If dt or n_steps carry predictive signal it means the integrator
    parameters co-vary with trajectory outcome — a leakage path that
    bypasses all early-window physics. A balanced accuracy above 0.50
    on these columns alone indicates the outcome is partially determined
    by the simulation setup, not the physical initial conditions.

    Returns
    -------
    dict with keys:
        warning_level : "none" | "OK" | "WARNING" | "CRITICAL"
        balanced_acc  : float (absent if no meta columns found)
    """
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    from ml_predictor.features import get_labels

    meta_feature_candidates = ["dt", "n_steps"]
    available = [c for c in meta_feature_candidates if c in df.columns]

    if len(available) == 0:
        if verbose:
            print("  [leakage_check] No meta columns found — skipping.")
        return {"warning_level": "none"}

    X_meta = df[available].values
    y      = get_labels(df).values

    model = lgb.LGBMClassifier(
        n_estimators=100,
        objective="multiclass",
        num_class=3,
        verbose=-1,
        random_state=42,
    )

    scores  = cross_val_score(model, X_meta, y, cv=5, scoring="balanced_accuracy")
    mean_ba = float(scores.mean())

    if mean_ba > 0.50:
        level = "CRITICAL"
    elif mean_ba > 0.40:
        level = "WARNING"
    else:
        level = "OK"

    if verbose:
        print(f"  [leakage_check] BA={mean_ba:.4f} | {level}")
    if level == "CRITICAL":
        print("  ⚠  CRITICAL LEAKAGE — fix early window before proceeding.")

    return {"balanced_acc": mean_ba, "warning_level": level}


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(report: EvaluationReport) -> None:
    """
    Compact single-block evaluation summary.

    Prints pooled OOF aggregate metrics and per-class F1.
    Regime breakdown is omitted from terminal output — it is preserved
    in the EvaluationReport object for downstream analysis.
    """
    label = (f"{report.model_name} | group {report.feature_group}"
             if report.feature_group else report.model_name)

    print(f"\n  [{label}]")
    print(f"  Macro F1: {report.macro_f1:.4f} (fold σ={report.fold_macro_f1_std:.4f}) | "
          f"Bal.Acc: {report.balanced_acc:.4f} | "
          f"Brier: {report.mean_brier:.4f}")
    print(f"  Per-class F1 — "
          + "  ".join(f"{k}: {v:.4f}" for k, v in report.per_class_f1.items()))