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

    Two F1 estimates are stored and are intentionally different:
      macro_f1          — pooled OOF: all held-out predictions concatenated
                          across folds, then F1 computed once. This is the
                          canonical publication number. Preferred on imbalanced
                          datasets because fold sizes differ and minority class
                          representation is uneven across folds.
      fold_macro_f1_mean — mean of per-fold F1s from CVResults.fold_macro_f1.
                           Useful as a stability cross-check. A gap > 0.01
                           relative to macro_f1 signals fold composition
                           variation worth investigating (see check_f1_discrepancy).
      fold_macro_f1_std  — fold-level variance; stability diagnostic independent
                           of the point estimate.
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
# FIX 5 — F1 DISCREPANCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_f1_discrepancy(report: EvaluationReport, threshold: float = 0.01) -> dict:
    """
    Check and report the gap between pooled OOF macro F1 and mean per-fold F1.

    Background
    ----------
    Two estimators of macro F1 exist after CV:
      (a) Pooled OOF F1 — all n held-out predictions stacked, F1 computed once.
          Canonical. Less sensitive to fold-size variance. Used in all_results.csv
          as macro_f1 and in published tables.
      (b) Mean per-fold F1 — arithmetic mean of the 5 fold-level F1 scores.
          Each fold's F1 weights all classes equally regardless of fold size.
          Can diverge from (a) when minority classes (chaotic in hierarchical
          regime, unstable in compact_equal) appear disproportionately in some
          folds.

    A gap > 0.01 is physically meaningful: it means the model's apparent
    performance depends on how the minority class happens to land in each fold.
    This is not a bug — it is a signal that StratifiedKFold's class-level
    stratification does not control for regime composition, and fold-level
    chaotic fractions vary enough to move F1 by > 0.01.

    Parameters
    ----------
    report    : EvaluationReport produced by evaluate().
    threshold : Gap size that triggers a warning. Default 0.01.

    Returns
    -------
    dict with keys:
        pooled_oof_f1   : float — canonical number (report.macro_f1)
        fold_mean_f1    : float — mean of per-fold F1s (report.fold_macro_f1_mean)
        gap             : float — abs(pooled - fold_mean)
        flagged         : bool  — True if gap > threshold
        canonical       : str   — always "pooled_oof" (for downstream clarity)
    """
    gap     = abs(report.macro_f1 - report.fold_macro_f1_mean)
    flagged = gap > threshold

    label = (f"{report.model_name} | group {report.feature_group}"
             if report.feature_group else report.model_name)

    print(f"  [F1 discrepancy | {label}]")
    print(f"    Pooled OOF macro F1  : {report.macro_f1:.4f}  ← canonical (published)")
    print(f"    Mean per-fold macro F1: {report.fold_macro_f1_mean:.4f}  ← stability cross-check")
    print(f"    Gap                  : {gap:.4f}", end="")

    if flagged:
        print(f"  ⚠  GAP > {threshold} — fold composition varies "
              f"(regime-structured minority class distribution likely cause)")
    else:
        print(f"  ✓  within tolerance")

    return {
        "pooled_oof_f1": report.macro_f1,
        "fold_mean_f1":  report.fold_macro_f1_mean,
        "gap":           round(gap, 4),
        "flagged":       flagged,
        "canonical":     "pooled_oof",
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — REGIME SAMPLE COUNT GUARD
# ─────────────────────────────────────────────────────────────────────────────

def check_regime_sample_counts(
    df:                   pd.DataFrame,
    min_samples_per_fold: int  = 5,
    n_folds:              int  = 5,
    verbose:              bool = True,
) -> dict:
    """
    Check per-regime per-class sample counts and flag combinations where
    n / n_folds < min_samples_per_fold.

    Physical motivation
    -------------------
    StratifiedKFold stratifies on outcome_class, not regime. In the real
    dataset, compact_equal has ~11 total samples — with 5-fold CV each fold
    receives ~2 compact_equal examples. At this scale, confusion matrix
    panels are degenerate: a single misclassification changes a recall value
    by 50%. Results for flagged regimes should be labelled "(low-n, interpret
    with caution)" in figures and excluded from quantitative claims.

    Parameters
    ----------
    df                   : Full dataset DataFrame with 'regime' and
                           'outcome_class' columns.
    min_samples_per_fold : Minimum per-regime per-class count expected per fold.
                           Default 5 (anything below makes CV statistics unreliable).
    n_folds              : CV folds — used to compute expected per-fold count.
    verbose              : Print the flagged combinations.

    Returns
    -------
    dict with keys:
        flagged_regimes : list[str] — regimes with at least one flagged class
        counts          : pd.DataFrame — n_samples per (regime, outcome_class)
        any_flagged     : bool
    """
    if "regime" not in df.columns or "outcome_class" not in df.columns:
        if verbose:
            print("  [regime_count_check] Missing 'regime' or 'outcome_class' — skipping.")
        return {"flagged_regimes": [], "counts": pd.DataFrame(), "any_flagged": False}

    counts = (
        df.groupby(["regime", "outcome_class"])
        .size()
        .reset_index(name="n_samples")
    )
    counts["n_per_fold"]  = counts["n_samples"] / n_folds
    counts["flagged"]     = counts["n_per_fold"] < min_samples_per_fold
    counts["class_name"]  = counts["outcome_class"].map(
        {0: "stable", 1: "unstable", 2: "chaotic"}
    )

    flagged_rows    = counts[counts["flagged"]]
    flagged_regimes = sorted(flagged_rows["regime"].unique().tolist())
    any_flagged     = len(flagged_regimes) > 0

    if verbose:
        if not any_flagged:
            print(f"  [regime_count_check] All regime×class combinations have "
                  f"≥ {min_samples_per_fold} samples/fold ✓")
        else:
            print(f"\n  ⚠  [regime_count_check] Low-n regime×class combinations "
                  f"(< {min_samples_per_fold} samples/fold):")
            for _, row in flagged_rows.iterrows():
                print(f"     {row['regime']:<18} | {row['class_name']:<10} | "
                      f"n={row['n_samples']:4d}  (~{row['n_per_fold']:.1f}/fold)")
            print(f"  Flagged regimes: {flagged_regimes}")
            print(f"  CV statistics for these regime×class combinations are unreliable.")
            print(f"  Confusion matrix panels will be annotated with ⚠ low-n.")

    return {
        "flagged_regimes": flagged_regimes,
        "counts":          counts,
        "any_flagged":     any_flagged,
    }


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
    feature_group : Feature group label (A–D) — stored on report for CSV.
    verbose       : Print compact one-block summary to stdout, including
                    F1 discrepancy check (pooled OOF vs mean per-fold).

    Returns
    -------
    EvaluationReport with all metrics and identity fields populated.
    """
    y_true  = results.y_true
    y_pred  = results.y_pred
    y_proba = results.y_proba

    # ── Overall OOF metrics ───────────────────────────────────────────────────
    # Pooled OOF computation — canonical metric for all published tables.
    # More reliable than mean(per-fold F1) on imbalanced datasets because
    # fold sizes differ and minority classes may be represented unevenly.
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
        # Fix 5: F1 discrepancy check — mandatory when verbose to surface
        # any fold-composition variance before results are written.
        check_f1_discrepancy(report)

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

            # Primary ranking metric — pooled OOF, canonical publication number
            "macro_f1":       round(r.macro_f1,      4),
            "macro_f1_std":   round(r.fold_macro_f1_std, 4),
            "balanced_acc":   round(r.balanced_acc,  4),
            "mean_brier":     round(r.mean_brier,    4),

            # Fold-level mean — cross-check against pooled OOF (see check_f1_discrepancy)
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
# FIX 6 — TWO-STAGE RESULTS PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_two_stage_csv(
    rows:       list[dict],
    output_dir: Path | str = Path("results/baseline"),
) -> pd.DataFrame:
    """
    Persist two-stage vs direct LightGBM comparison to two_stage_results.csv.

    Each row in `rows` corresponds to one feature group and must contain:
        feature_group, direct_ba, s1_ba, s2_ba, direct_f1, s1_f1, s2_f1

    The function derives two summary columns:
        two_stage_wins_ba  : bool — combined stage ba > direct ba
        two_stage_wins_f1  : bool — stage2 f1 > direct f1
    and prints one auto-generated sentence per group for the paper.

    Physical interpretation
    -----------------------
    Stage 1 (stable vs not-stable) separates geometry-driven instability
    (Hill criterion, mass ratios, separation). Stage 2 (unstable vs chaotic)
    separates energy-exchange dynamics (encounter depth, eccentricity growth).
    If the two-stage model outperforms direct classification on balanced
    accuracy, it suggests these two physical boundaries are qualitatively
    different enough that conflating them in a single 3-class model loses
    information. If the direct model wins, the feature set already encodes
    both boundaries jointly.

    Parameters
    ----------
    rows       : List of dicts accumulated in run_baseline's baseline loop.
    output_dir : Output directory.

    Returns
    -------
    DataFrame written to {output_dir}/two_stage_results.csv.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Combined two-stage balanced accuracy: harmonic mean of stage1 and stage2
    # (both stages must perform well — a good stage1 cannot compensate for a
    # degenerate stage2 that only sees the not-stable subset).
    df["two_stage_combined_ba"] = 2 * df["s1_ba"] * df["s2_ba"] / (
        df["s1_ba"] + df["s2_ba"] + 1e-12
    )
    df["two_stage_wins_ba"] = df["two_stage_combined_ba"] > df["direct_ba"]
    df["two_stage_wins_f1"] = df["s2_f1"] > df["direct_f1"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "two_stage_results.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # ── Auto-generated interpretation sentences ───────────────────────────────
    print("\n  Two-stage vs direct — interpretation:")
    for _, row in df.iterrows():
        fg = row["feature_group"]
        if row["two_stage_wins_ba"]:
            sentence = (
                f"Group {fg}: two-stage LightGBM (stage1 BA={row['s1_ba']:.3f}, "
                f"stage2 BA={row['s2_ba']:.3f}) outperforms direct classification "
                f"(BA={row['direct_ba']:.3f}), supporting the physical separation "
                f"of the Hill-criterion and energy-exchange instability boundaries."
            )
        else:
            sentence = (
                f"Group {fg}: direct LightGBM (BA={row['direct_ba']:.3f}) matches or "
                f"exceeds two-stage classification (combined BA={row['two_stage_combined_ba']:.3f}), "
                f"suggesting the {fg} feature set encodes both instability boundaries jointly."
            )
        print(f"  → {sentence}")

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

    Prints pooled OOF macro F1 (canonical) alongside fold mean for immediate
    discrepancy visibility. Regime breakdown is omitted from terminal output —
    it is preserved in the EvaluationReport object for downstream analysis.
    """
    label = (f"{report.model_name} | group {report.feature_group}"
             if report.feature_group else report.model_name)

    print(f"\n  [{label}]")
    print(f"  Macro F1 (pooled OOF): {report.macro_f1:.4f} | "
          f"Fold mean: {report.fold_macro_f1_mean:.4f} | "
          f"Fold σ: {report.fold_macro_f1_std:.4f}")
    print(f"  Bal.Acc: {report.balanced_acc:.4f} | "
          f"Brier: {report.mean_brier:.4f}")
    print(f"  Per-class F1 — "
          + "  ".join(f"{k}: {v:.4f}" for k, v in report.per_class_f1.items()))


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 2 + FIX 7 — CV STABILITY GUARD
# ─────────────────────────────────────────────────────────────────────────────

def flag_unstable_runs(
    df_results:     pd.DataFrame,
    std_threshold:  float = 0.015,
    verbose:        bool  = True,
) -> pd.DataFrame:
    """
    Scan a results DataFrame for runs whose fold-level macro F1 std exceeds
    the publication threshold and mark them as excluded. Also excludes
    CatBoost C-group runs (Fix 7).

    Two exclusion criteria are applied:

    1. std-based (Item 2): any row with macro_f1_std > std_threshold is flagged.
       CatBoost cross-validation std on groups B20–B30 exceeds 0.030–0.040
       under default hyperparameters — fold-level instability driven by
       CatBoost's symmetric tree structure interacting with regime-structured
       class imbalance. The threshold 0.015 gives CatBoost a 2× margin vs
       LightGBM/XGBoost (which consistently achieve std ≤ 0.008).

    2. CatBoost C-group (Fix 7): CatBoost on C-groups (window-only, 7 features)
       is excluded editorially. Seven window-only features give symmetric trees
       nothing physically meaningful to split on — the IC geometry (hill_ratio,
       r3_sep/r12_init, mass ratios) that sets the instability threshold is
       entirely absent. Results are near-random and pollute the comparison table.
       Excluded regardless of std.

    Parameters
    ----------
    df_results    : DataFrame produced by save_results_csv — must contain
                    columns "model", "feature_group", "macro_f1_std".
    std_threshold : Flag any row with macro_f1_std > this value.
                    Default 0.015 (PDF Item 2 requirement).
    verbose       : Print a summary of flagged rows to stdout.

    Returns
    -------
    DataFrame — copy of df_results with two new columns:
        "stable_run"  : bool — True if both criteria pass (safe to publish).
        "exclude_note": str  — empty for stable runs; reason string for flagged.
    """
    df = df_results.copy()

    if "macro_f1_std" not in df.columns:
        raise ValueError(
            "flag_unstable_runs requires a 'macro_f1_std' column. "
            "Ensure save_results_csv has been called first."
        )

    # Criterion 1: fold std exceeds threshold
    std_mask = df["macro_f1_std"] > std_threshold

    # Criterion 2: CatBoost on window-only (C-group) — editorial exclusion
    cgroup_mask = (
        (df["model"] == "catboost") &
        (df["feature_group"].str.upper().str.startswith("C"))
    )

    unstable_mask = std_mask | cgroup_mask

    df["stable_run"]   = ~unstable_mask
    df["exclude_note"] = ""

    # Apply std-based note first, then override/append for C-group where both fire
    df.loc[std_mask, "exclude_note"] = (
        f"CV std > {std_threshold:.3f} — fold-level instability; "
        "retune or exclude from comparison table"
    )
    df.loc[cgroup_mask, "exclude_note"] = (
        "CatBoost C-group excluded: window-only features (7) insufficient "
        "for symmetric tree splitting — no IC geometry, results near-random"
    )
    # Where both apply, concatenate
    both_mask = std_mask & cgroup_mask
    if both_mask.any():
        df.loc[both_mask, "exclude_note"] = (
            f"CV std > {std_threshold:.3f} AND CatBoost C-group (window-only, "
            "no IC geometry)"
        )

    if verbose:
        n_std     = std_mask.sum()
        n_cgroup  = cgroup_mask.sum()
        n_flagged = unstable_mask.sum()

        if n_flagged == 0:
            print(f"  [stability_guard] All {len(df)} runs pass exclusion criteria ✓")
        else:
            print(f"\n  ⚠  [stability_guard] {n_flagged} run(s) excluded:")
            if n_std > 0:
                print(f"     Criterion 1 (std > {std_threshold}): {n_std} run(s)")
                flagged_std = df[std_mask][["model", "feature_group",
                                            "macro_f1", "macro_f1_std"]]
                for _, row in flagged_std.iterrows():
                    print(f"       {row['model']:<12} | group {row['feature_group']:<5} | "
                          f"F1={row['macro_f1']:.4f}  std={row['macro_f1_std']:.4f}")
            if n_cgroup > 0:
                cgroups = df[cgroup_mask]["feature_group"].tolist()
                print(f"     Criterion 2 (CatBoost C-groups): {n_cgroup} run(s) — "
                      f"{cgroups}")
            print(f"  Marked stable_run=False. Do NOT include in published table.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 6 — EXPECTED CALIBRATION ERROR
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    y_true_binary: np.ndarray,
    y_prob:        np.ndarray,
    n_bins:        int = 10,
) -> float:
    """
    Expected Calibration Error — weighted mean |observed_fraction - predicted_prob|
    across equal-width probability bins.

    ECE measures the average discrepancy between predicted confidence and
    empirical frequency. For the chaotic class in three-body classification,
    ECE is particularly informative: a model that assigns P(chaotic)=0.4 when
    the true fraction is 0.55 is systematically under-confident in the
    moderate-probability regime. This is consequential for computational
    triage — systems assigned P(chaotic)=0.4 would be deprioritised for
    further integration when they should not be.

    This function operates on whatever array is passed — it makes no
    assumption about whether the input comes from a train/test split,
    a full OOF array, or a calibration holdout subset. The caller is
    responsible for passing the correct subset (e.g. the 20% OOF holdout
    used in plot_calibration_before_after for the post-calibration ECE).

    Parameters
    ----------
    y_true_binary : Binary array — 1 where true class == target class, else 0.
    y_prob        : Predicted probability for the target class (same length).
    n_bins        : Number of equal-width bins in [0, 1]. Default 10.

    Returns
    -------
    ece : float — lower is better. Well calibrated ≈ 0.

    Interpretation thresholds (from Action List Item 6)
    ---------------------------------------------------
    < 0.05          : Well calibrated — probabilities usable directly for triage.
    0.05 – 0.10     : Moderate bias — report ECE; isotonic regression recommended.
    > 0.10          : Poorly calibrated — report as limitation even after correction.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true_binary)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        observed_frac = float(y_true_binary[mask].mean())
        mean_conf     = float(y_prob[mask].mean())
        ece += mask.sum() * abs(observed_frac - mean_conf)

    return ece / n


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 4 — MEGNO BORDERLINE AUDIT
# ─────────────────────────────────────────────────────────────────────────────

# Threshold: if this fraction of chaotic-stable confusions falls within
# [megno_threshold, megno_threshold + 1.0], the errors are attributed to
# labelling ambiguity rather than genuine classification failure. Set at 0.60 —
# majority concentration near the threshold is sufficient to conclude the model
# is not making avoidable errors on clearly chaotic systems.
_LABEL_AMBIGUITY_FRACTION_THRESHOLD = 0.60


def _make_megno_bins(megno_threshold: float) -> list[tuple[float, float]]:
    """
    Build MEGNO bin edges dynamically from a given labelling threshold.

    The first bin starts at megno_threshold (the labelling boundary) and
    spans 0.5 units. Subsequent bins widen to capture the heavy right tail
    of MEGNO distributions. The final bin is open-ended (inf).

    Parametrising bins from the threshold ensures the "near-threshold" zone
    [megno_threshold, megno_threshold + 1.0] is always captured by exactly
    the first two bins, regardless of threshold value.
    """
    t = megno_threshold
    return [
        (t,       t + 0.5),
        (t + 0.5, t + 1.0),
        (t + 1.0, t + 2.0),
        (t + 2.0, t + 7.0),
        (t + 7.0, np.inf),
    ]


def megno_borderline_audit(
    oof_df:          pd.DataFrame,
    megno_threshold: float = 3.0,
    verbose:         bool  = True,
) -> pd.DataFrame:
    """
    Audit whether chaotic-to-stable misclassifications are driven by labelling
    ambiguity (MEGNO barely above threshold) or genuine classification errors
    (MEGNO spread across a wide range).

    This is a post-hoc diagnostic on existing OOF predictions — it does NOT
    require retraining. MEGNO is never used as a feature (it is in LEAKY_COLS);
    it is only joined to OOF predictions here for interpretive purposes.

    Physical motivation
    -------------------
    MEGNO converges to ≈2 for quasi-periodic orbits and diverges linearly for
    chaotic ones, but convergence is slow — O(T) integration time is required
    for reliable discrimination near the chaos boundary. A system with MEGNO
    barely above megno_threshold after finite integration may represent:
      (a) Genuine slow chaos: Lyapunov exponent is small but positive; longer
          integration would push MEGNO higher. The model cannot detect this from
          30% of T_inner — this is a fundamental limit of early-time prediction.
      (b) Labelling noise: the integration was too short, MEGNO has not converged,
          and the system might actually be quasi-periodic.
    The audit distinguishes these scenarios by checking whether confused systems
    concentrate near the threshold (case a/b — ambiguity) or spread to high MEGNO
    (genuine model failure).

    Fix 3: megno_threshold is now a parameter (default 3.0). Run at 3.0, 4.0,
    and 5.0 to test sensitivity — if the labelling ambiguity interpretation holds,
    the near-threshold concentration should persist at higher thresholds because
    the confusion pattern should track the boundary, not the absolute MEGNO value.
    MEGNO bins are generated dynamically from megno_threshold via _make_megno_bins.

    Parameters
    ----------
    oof_df          : DataFrame with columns:
                        "true_class" : int  (0=stable, 1=unstable, 2=chaotic)
                        "pred_class" : int
                        "MEGNO"      : float — raw MEGNO value (NOT MEGNO_clean)
                        "regime"     : str   (optional but recommended)
                      Produced by run_baseline.py when --megno-audit flag is set.
    megno_threshold : MEGNO value used as the labelling boundary. Default 3.0
                      (dataset labelling threshold). Pass 4.0 or 5.0 for
                      sensitivity analysis (Fix 3).
    verbose         : Print per-bin breakdown and interpretation sentence.

    Returns
    -------
    audit_df : DataFrame — one row per MEGNO bin with columns:
                 "megno_lo", "megno_hi",
                 "n_chaotic_total"    : all true-chaotic systems in this bin
                 "n_confused_stable"  : true-chaotic predicted stable
                 "n_confused_unstable": true-chaotic predicted unstable
                 "n_correct"          : true-chaotic predicted chaotic
                 "pct_confused_stable": confused_stable / total (%)
               attrs["interpretation"]       : interpretation sentence
               attrs["frac_near_threshold"]  : fraction near [t, t+1.0]
               attrs["megno_threshold_used"] : threshold passed in

    Side effects
    ------------
    When verbose=True, prints:
      - Median MEGNO for each prediction outcome
      - Per-bin breakdown table
      - Regime distribution of chaotic-stable confusions
      - One interpretation sentence (labelling ambiguity vs genuine error)
    """
    required = {"true_class", "pred_class", "MEGNO"}
    missing  = required - set(oof_df.columns)
    if missing:
        raise ValueError(
            f"megno_borderline_audit: oof_df is missing columns: {missing}. "
            f"Ensure the DataFrame was produced with MEGNO joined from the "
            f"metadata CSV (not from feature engineering)."
        )

    # Redefine "true chaotic" relative to the requested threshold.
    # At threshold=4.0, only systems with MEGNO >= 4.0 are treated as
    # unambiguously chaotic for the purpose of this audit.
    true_chaotic     = oof_df[oof_df["MEGNO"] >= megno_threshold]
    # For prediction subsets, keep the original true_class labels —
    # we are auditing prediction errors on high-MEGNO systems.
    chaotic_stable   = true_chaotic[true_chaotic["pred_class"] == 0]
    chaotic_unstable = true_chaotic[true_chaotic["pred_class"] == 1]
    chaotic_correct  = true_chaotic[true_chaotic["pred_class"] == 2]

    if verbose:
        print(f"\n── MEGNO Borderline Audit (threshold={megno_threshold:.1f}) "
              f"─────────────────────────────")
        print(f"  Systems with MEGNO ≥ {megno_threshold:.1f}  "
              f"(n={len(true_chaotic):5d}): "
              f"median MEGNO = {true_chaotic['MEGNO'].median():.2f}")
        for subset, label in [
            (chaotic_stable,   "Pred stable  "),
            (chaotic_unstable, "Pred unstable"),
            (chaotic_correct,  "Pred correct "),
        ]:
            if len(subset) > 0:
                print(f"  {label} (n={len(subset):5d}): "
                      f"median MEGNO = {subset['MEGNO'].median():.2f}")
            else:
                print(f"  {label} (n=    0): median MEGNO = N/A")

    # ── Per-bin breakdown ─────────────────────────────────────────────────────
    megno_bins = _make_megno_bins(megno_threshold)
    rows = []

    for lo, hi in megno_bins:
        in_bin            = true_chaotic[(true_chaotic["MEGNO"] >= lo) & (true_chaotic["MEGNO"] < hi)]
        confused_stable   = chaotic_stable[(chaotic_stable["MEGNO"] >= lo) & (chaotic_stable["MEGNO"] < hi)]
        confused_unstable = chaotic_unstable[(chaotic_unstable["MEGNO"] >= lo) & (chaotic_unstable["MEGNO"] < hi)]
        correct           = chaotic_correct[(chaotic_correct["MEGNO"] >= lo) & (chaotic_correct["MEGNO"] < hi)]

        n_total = len(in_bin)
        n_cs    = len(confused_stable)
        n_cu    = len(confused_unstable)
        n_ok    = len(correct)
        pct_cs  = (n_cs / n_total * 100) if n_total > 0 else 0.0

        rows.append({
            "megno_lo":            lo,
            "megno_hi":            hi if hi != np.inf else 9999.0,
            "n_chaotic_total":     n_total,
            "n_confused_stable":   n_cs,
            "n_confused_unstable": n_cu,
            "n_correct":           n_ok,
            "pct_confused_stable": round(pct_cs, 1),
        })

        if verbose and n_total > 0:
            hi_str = f"{hi:.1f}" if hi != np.inf else "∞"
            print(f"  MEGNO [{lo:.1f}, {hi_str}): "
                  f"n={n_total:4d} total | "
                  f"{n_cs:3d} confused→stable ({pct_cs:4.1f}%) | "
                  f"{n_cu:3d} confused→unstable | "
                  f"{n_ok:3d} correct")

    audit_df = pd.DataFrame(rows)

    # ── Regime breakdown of chaotic-stable confusions ─────────────────────────
    if verbose and "regime" in oof_df.columns and len(chaotic_stable) > 0:
        print("\n  Chaotic→Stable confusion by regime:")
        regime_counts = chaotic_stable["regime"].value_counts()
        total_cs      = len(chaotic_stable)
        for regime, cnt in regime_counts.items():
            print(f"    {regime:<18}: {cnt:4d} ({cnt/total_cs*100:.1f}%)")

    # ── Interpretation ────────────────────────────────────────────────────────
    # Fraction of chaotic-stable confusions within [threshold, threshold + 1.0].
    # This is the "near-threshold zone" where labelling ambiguity is plausible.
    n_total_cs  = len(chaotic_stable)
    near_thresh = chaotic_stable[
        (chaotic_stable["MEGNO"] >= megno_threshold) &
        (chaotic_stable["MEGNO"] < megno_threshold + 1.0)
    ]
    frac_near = len(near_thresh) / n_total_cs if n_total_cs > 0 else 0.0

    if frac_near >= _LABEL_AMBIGUITY_FRACTION_THRESHOLD:
        interpretation = (
            f"LABELLING AMBIGUITY (threshold={megno_threshold:.1f}): "
            f"The chaotic→stable confusion is concentrated in systems with "
            f"MEGNO in [{megno_threshold:.1f}, {megno_threshold+1.0:.1f}] — "
            f"the threshold boundary ({frac_near*100:.0f}% of confused systems). "
            f"These represent systems where Lyapunov divergence is not yet manifest "
            f"at 30% T_inner, consistent with the finite convergence timescale of MEGNO."
        )
    else:
        interpretation = (
            f"GENUINE CLASSIFICATION ERROR (threshold={megno_threshold:.1f}): "
            f"A fraction of chaotic systems are misclassified independent of their "
            f"MEGNO magnitude (only {frac_near*100:.0f}% near threshold "
            f"[{megno_threshold:.1f}, {megno_threshold+1.0:.1f}]). This suggests "
            f"the stable-chaotic boundary requires dynamical information beyond "
            f"30% of the inner orbital period."
        )

    if verbose:
        print(f"\n  Interpretation ({frac_near*100:.0f}% near threshold "
              f"[{megno_threshold:.1f}, {megno_threshold+1.0:.1f}]):")
        print(f"  → {interpretation}")
        print("─" * 63)

    audit_df.attrs["interpretation"]       = interpretation
    audit_df.attrs["frac_near_threshold"]  = round(frac_near, 4)
    audit_df.attrs["megno_threshold_used"] = megno_threshold

    return audit_df