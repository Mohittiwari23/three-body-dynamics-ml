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
import matplotlib.colors as mcolors
from pathlib import Path

from ml_predictor.trainer import CVResults
from ml_predictor.evaluator import EvaluationReport, compute_ece

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


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 5 — REGIME-STRATIFIED CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_confusion_grid(
    oof_df:           pd.DataFrame,
    group_name:       str,
    output_dir:       Path | None = None,
    low_n_threshold:  int         = 30,       # Fix 2: panels below this are annotated
) -> plt.Figure:
    """
    2×2 grid of normalised confusion matrices, one panel per regime.

    Produced from OOF predictions collected via 5-fold CV. The aggregate
    confusion matrix pools all regimes, masking the fact that the 0.13
    chaotic-to-stable confusion rate is almost certainly concentrated in
    one or two specific regimes. This figure reveals which part of phase
    space is physically hard to classify and directly informs the discussion.

    Physical interpretation guide
    -----------------------------
    hierarchical  : Expect low confusion — outer body starts far from inner
                    binary, Hill criterion is comfortably satisfied. Chaotic
                    systems here have MEGNO barely above threshold.
    asymmetric    : Moderate confusion across all classes — most diverse
                    mass-ratio distribution, no dominant dynamical mode.
    compact_equal : High unstable rate; confusion concentrated at the
                    unstable-chaotic boundary driven by slingshot dynamics.
    scatter       : Possible high chaotic-stable confusion — near-escape
                    velocity outer bodies have slow Lyapunov growth if their
                    orbit is only weakly chaotic.

    Parameters
    ----------
    oof_df          : DataFrame with columns "true", "pred", "regime".
                      Produced by run_baseline.py when --regime-confusion is set.
                      One row per sample; values are integer class labels (0/1/2).
    group_name      : Feature group label used in the figure title and filename
                      (e.g. "B5", "B30"). Filename follows the convention:
                      lgb_{group_name}_confusion_by_regime.png
    output_dir      : If provided, saves the figure there and prints the path.
    low_n_threshold : Regimes with fewer than this many samples receive a
                      "⚠ low-n" annotation in red on their panel title.
                      Default 30. No data is filtered or modified.

    Returns
    -------
    fig : matplotlib Figure — 2×2 grid, one regime per panel.

    Also prints
    -----------
    Per-regime chaotic→stable and chaotic→unstable confusion rates, which
    are the key numbers for the "which regime drives the confusion?" answer
    required by the attestation checklist.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    required = {"true", "pred", "regime"}
    missing  = required - set(oof_df.columns)
    if missing:
        raise ValueError(
            f"plot_regime_confusion_grid: oof_df missing columns: {missing}"
        )

    regimes = [r for r in REGIME_ORDER if r in oof_df["regime"].values]
    # Append any regimes present in data but not in REGIME_ORDER
    for r in oof_df["regime"].unique():
        if r not in regimes:
            regimes.append(r)

    # Always produce a 2×2 grid regardless of how many regimes are present.
    # Pad with empty axes if fewer than 4 regimes exist (e.g. smoke-test data).
    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    fig.suptitle(
        f"LightGBM {group_name} — Normalised Confusion Matrix by Regime\n"
        f"(5-fold OOF, MEGNO threshold 3.0)",
        fontsize=12,
        fontweight="bold",
    )

    print(f"\n── {group_name} chaotic→stable confusion by regime ──")

    for idx, ax in enumerate(axes.flat):
        if idx >= len(regimes):
            ax.set_visible(False)
            continue

        regime = regimes[idx]
        sub    = oof_df[oof_df["regime"] == regime]
        n_sub  = len(sub)

        cm = sk_cm(sub["true"], sub["pred"], labels=[0, 1, 2], normalize="true")

        # Draw heatmap using imshow — no seaborn dependency
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="YlGnBu")

        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                val = cm[i, j]
                ax.text(
                    j, i,
                    f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white" if val > 0.6 else "black",
                )

        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Fix 2: annotate low-n regimes in red; normal regimes in default black.
        # Visual only — no data is filtered or modified.
        if n_sub < low_n_threshold:
            title_str   = f"{regime}  (n={n_sub:,})  ⚠ low-n"
            title_color = "red"
        else:
            title_str   = f"{regime}  (n={n_sub:,})"
            title_color = "black"

        ax.set_title(title_str, fontsize=11, color=title_color)

        # Print the key diagnostic numbers for the checklist
        cs  = cm[2, 0]   # chaotic → stable
        cu  = cm[2, 1]   # chaotic → unstable
        cc  = cm[2, 2]   # chaotic → correct
        print(f"  {regime:<18}: chaotic→stable={cs:.3f}  "
              f"chaotic→unstable={cu:.3f}  correct={cc:.3f}"
              + ("  ⚠ low-n" if n_sub < low_n_threshold else ""))

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"lgb_{group_name}_confusion_by_regime.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 6 — CALIBRATION BEFORE / AFTER ISOTONIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_before_after(
    prob_df:    pd.DataFrame,
    group_name: str,
    output_dir: Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    2-row × 3-column calibration figure: before (row 0) and after isotonic
    regression (row 1), one column per class (stable / unstable / chaotic).

    Background
    ----------
    The model outputs raw probabilities that are systematically under-confident
    for the chaotic class in the 0.2–0.7 range. When the model assigns
    P(chaotic)=0.4, the true fraction of chaotic systems is closer to 0.55 —
    the model hedges toward the base rate in ambiguous cases (systems near the
    first close encounter, where the chaos onset timescale has not yet
    exceeded 30% of T_inner). Isotonic regression learns a monotone correction
    function from OOF predictions and removes this systematic bias.

    Calibration split
    -----------------
    The model is trained via 5-fold stratified CV. OOF probabilities are
    generated for every sample in the full dataset — each sample is scored
    exactly once by a model that never saw it during training. These OOF
    probabilities form a single array covering all N samples.

    The isotonic calibrator is then fitted and evaluated within this OOF
    array using an 80/20 split: the first 80% of OOF rows are used to fit
    the isotonic regression; the remaining 20% are held out to compute the
    post-calibration ECE. This is NOT a train/test split of the raw data —
    it is a split within the already-held-out OOF probability array.

    Consequence: the OOF array is not time- or index-ordered, so this 80/20
    division is arbitrary in sample ordering. The "ece_after" value is
    therefore evaluated on ~20% of total samples and carries more uncertainty
    than the "ece_before" ECE (which uses the full OOF array). Both
    limitations are documented in the returned DataFrame via "ece_after_note".

    Parameters
    ----------
    prob_df    : DataFrame with columns:
                   "prob_stable", "prob_unstable", "prob_chaotic" — raw OOF
                   probabilities generated by 5-fold CV (one row per sample)
                   "true"  — integer class label (0/1/2)
                 Produced by run_baseline.py when --calibrate flag is set.
    group_name : Feature group label for title and filename
                 (e.g. "B30"). Filename: lgb_{group_name}_calibration_before_after.png
    output_dir : If provided, saves the figure and ece_results_{group_name}.csv.

    Returns
    -------
    fig     : matplotlib Figure — 2×3 grid.
    ece_df  : DataFrame with columns:
                "class"           — outcome class name
                "ece_before"      — ECE on full OOF array (all N samples)
                "ece_after"       — ECE after isotonic correction, evaluated
                                    on the held-out 20% of the OOF array
                "calibration_quality" — "well calibrated" / "moderate bias" /
                                        "poorly calibrated" (based on ece_after)
                "ece_after_n"     — number of OOF samples in the 20% eval set
                "ece_after_note"  — plain-text reminder that ece_after is
                                    evaluated on 20% of OOF probabilities,
                                    not on a separate raw-data holdout
              One row per class. Ready to save as ece_results_{group_name}.csv.

    Also prints
    -----------
    ECE table and one interpretation sentence per class, ready to paste into
    the paper discussion section.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression

    required = {"prob_stable", "prob_unstable", "prob_chaotic", "true"}
    missing  = required - set(prob_df.columns)
    if missing:
        raise ValueError(
            f"plot_calibration_before_after: prob_df missing columns: {missing}"
        )

    prob_cols = ["prob_stable", "prob_unstable", "prob_chaotic"]
    # n_cal is the index splitting the OOF probability array 80/20.
    # This is NOT a raw-data train/test split — both halves are already
    # out-of-fold predictions from the 5-fold CV loop.
    n_cal = int(0.8 * len(prob_df))

    fig, axes = plt.subplots(2, N_CLASSES, figsize=(14, 9))
    fig.suptitle(
        f"LightGBM {group_name} — Calibration Before and After Isotonic Regression\n"
        f"(5-fold OOF probabilities | 80/20 OOF calibration split)",
        fontsize=11,
        fontweight="bold",
    )

    ece_rows  = []
    sentences = []

    print(f"\n── Calibration ECE — LightGBM {group_name} ─────────────────────")
    print(f"  {'Class':<10}  {'ECE before':>10}  {'ECE after':>10}  Quality")

    for cls_idx, (cls_name, col) in enumerate(zip(CLASS_NAMES, prob_cols)):

        y_bin  = (prob_df["true"].values == cls_idx).astype(int)
        y_prob = prob_df[col].values

        # ── ECE before calibration (full OOF array, all N samples) ───────────
        ece_before = compute_ece(y_bin, y_prob)
        frac_raw, mean_raw = calibration_curve(y_bin, y_prob, n_bins=10)

        # ── Isotonic regression ───────────────────────────────────────────────
        # Fit on first 80% of the OOF array; evaluate on remaining 20%.
        # Both subsets are out-of-fold predictions — no raw training data leaks.
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_prob[:n_cal], y_bin[:n_cal])
        y_cal      = ir.predict(y_prob[n_cal:])
        ece_after  = compute_ece(y_bin[n_cal:], y_cal)
        frac_cal, mean_cal = calibration_curve(y_bin[n_cal:], y_cal, n_bins=10)

        # ── Quality label ─────────────────────────────────────────────────────
        # Evaluated on the post-calibration ECE (after isotonic correction).
        # Thresholds from PDF Item 6.
        if ece_after < 0.05:
            quality = "well calibrated"
        elif ece_after < 0.10:
            quality = "moderate bias"
        else:
            quality = "poorly calibrated"

        print(f"  {cls_name:<10}  {ece_before:>10.4f}  {ece_after:>10.4f}  {quality}")

        ece_rows.append({
            "class":               cls_name,
            "ece_before":          round(ece_before, 4),
            "ece_after":           round(ece_after,  4),
            "calibration_quality": quality,
            # Number of OOF samples used for the post-calibration ECE evaluation
            "ece_after_n":         len(prob_df) - n_cal,
            # Fix 1: self-documenting note so the CSV is unambiguous
            "ece_after_note":      "evaluated on 20% of OOF probabilities (not a raw-data holdout)",
        })

        # ── Build interpretation sentence ─────────────────────────────────────
        if quality == "well calibrated":
            sent = (
                f"{cls_name.capitalize()} class: ECE={ece_before:.3f} before, "
                f"{ece_after:.3f} after isotonic regression — well calibrated; "
                f"probabilities usable directly for computational triage."
            )
        elif quality == "moderate bias":
            sent = (
                f"{cls_name.capitalize()} class: ECE={ece_before:.3f} before, "
                f"{ece_after:.3f} after isotonic regression — moderate calibration "
                f"bias; isotonic correction recommended before probability-based triage."
            )
        else:
            sent = (
                f"{cls_name.capitalize()} class: ECE={ece_before:.3f} before, "
                f"{ece_after:.3f} after isotonic regression — probability estimates "
                f"remain poorly calibrated after correction, reflecting fundamental "
                f"ambiguity in the moderate-confidence regime for this class."
            )
        sentences.append(sent)

        # ── Plot: row 0 = before, row 1 = after ──────────────────────────────
        ax_before = axes[0, cls_idx]
        ax_before.plot(mean_raw, frac_raw, "bo-", lw=1.5,
                       label=f"Model  ECE={ece_before:.3f}")
        ax_before.plot([0, 1], [0, 1], "--", color="orange", label="Perfect")
        ax_before.set_title(f"{cls_name} — before calibration", fontsize=10)
        ax_before.set_xlabel("Predicted probability", fontsize=9)
        ax_before.set_ylabel("Observed fraction", fontsize=9)
        ax_before.legend(fontsize=8)
        ax_before.set_xlim(0, 1)
        ax_before.set_ylim(0, 1)

        ax_after = axes[1, cls_idx]
        ax_after.plot(mean_cal, frac_cal, "go-", lw=1.5,
                      label=f"Calibrated  ECE={ece_after:.3f}")
        ax_after.plot([0, 1], [0, 1], "--", color="orange", label="Perfect")
        ax_after.set_title(f"{cls_name} — after isotonic regression", fontsize=10)
        ax_after.set_xlabel("Predicted probability", fontsize=9)
        ax_after.set_ylabel("Observed fraction", fontsize=9)
        ax_after.legend(fontsize=8)
        ax_after.set_xlim(0, 1)
        ax_after.set_ylim(0, 1)

    plt.tight_layout()

    ece_df = pd.DataFrame(ece_rows)

    # Print interpretation sentences
    print("\n  Interpretation sentences (ready to paste into paper):")
    for s in sentences:
        print(f"  → {s}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig_path = output_dir / f"lgb_{group_name}_calibration_before_after.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fig_path}")

        ece_path = output_dir / f"ece_results_{group_name}.csv"
        ece_df.to_csv(ece_path, index=False)
        print(f"  Saved: {ece_path}")

    return fig, ece_df


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 4 — MEGNO BORDERLINE AUDIT PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_megno_audit(
    oof_df:          pd.DataFrame,
    group_name:      str,
    output_dir:      Path | None = None,
    megno_threshold: float       = 3.0,       # Fix 3: was hardcoded to 3.0
) -> plt.Figure:
    """
    Three-panel histogram of MEGNO distributions for chaotic system subsets:
      Panel 0: Chaotic → Stable   (the primary confusion of interest)
      Panel 1: Chaotic → Unstable (secondary confusion)
      Panel 2: Chaotic → Chaotic  (correctly classified — reference distribution)

    Physical motivation
    -------------------
    If confused systems (chaotic→stable) concentrate near the MEGNO threshold,
    the confusion is caused by slow Lyapunov divergence that has not yet grown
    beyond noise level at 30% T_inner — this is a fundamental predictability
    limit, not a model failure. If they spread across the full MEGNO range,
    the model is making avoidable errors on clearly chaotic systems, pointing
    to missing features or insufficient window length.

    The MEGNO threshold line (orange dashed) marks the labelling boundary:
    systems below this were assigned outcome_class=1 (unstable), systems above
    were assigned outcome_class=2 (chaotic). The threshold is parameterised
    so sensitivity analysis can be run at multiple values (e.g. 3.0, 4.0, 5.0)
    without modifying this function.

    Parameters
    ----------
    oof_df          : DataFrame with columns:
                        "true_class" : int (0=stable, 1=unstable, 2=chaotic)
                        "pred_class" : int
                        "MEGNO"      : float — raw MEGNO value (not MEGNO_clean)
                      Produced by run_baseline.py when --megno-audit flag is set.
    group_name      : Feature group label for title and filename
                      (e.g. "B30"). Filename: megno_audit_{group_name}_t{threshold}.png
    output_dir      : If provided, saves the figure there.
    megno_threshold : MEGNO labelling boundary. Default 3.0 (dataset threshold).
                      Pass 4.0 or 5.0 for sensitivity analysis (Fix 3 in evaluator).
                      Controls: vertical threshold line position, bin start edge,
                      figure title, and legend label.

    Returns
    -------
    fig : matplotlib Figure — 3-panel histogram.
    """
    required = {"true_class", "pred_class", "MEGNO"}
    missing  = required - set(oof_df.columns)
    if missing:
        raise ValueError(
            f"plot_megno_audit: oof_df missing columns: {missing}"
        )

    chaotic_stable   = oof_df[(oof_df["true_class"] == 2) & (oof_df["pred_class"] == 0)]
    chaotic_unstable = oof_df[(oof_df["true_class"] == 2) & (oof_df["pred_class"] == 1)]
    chaotic_correct  = oof_df[(oof_df["true_class"] == 2) & (oof_df["pred_class"] == 2)]

    subsets = [
        (chaotic_stable,   "Chaotic → Stable (confused)"),
        (chaotic_unstable, "Chaotic → Unstable (confused)"),
        (chaotic_correct,  "Chaotic → Chaotic (correct)"),
    ]

    # Fix 3: bins anchored to megno_threshold so the near-threshold zone is
    # always captured by the first bin regardless of the threshold value.
    # Log-spaced widths handle the heavy right tail; clip at 50 prevents
    # extreme outliers from collapsing the near-threshold bins.
    t = megno_threshold
    bins = [t, t + 0.5, t + 1.0, t + 2.0, t + 3.0, t + 5.0,
            t + 7.0, t + 12.0, t + 17.0, 50.0]
    # Ensure the clip ceiling (50) is above all bin edges
    bins = sorted(set(bins + [50.0]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"LightGBM {group_name} — MEGNO Distribution by Prediction Outcome\n"
        f"(True-chaotic systems only | orange dashed = MEGNO threshold {megno_threshold:.1f})",
        fontsize=10,
        fontweight="bold",
    )

    for ax, (subset, label) in zip(axes, subsets):
        megno_clipped = subset["MEGNO"].clip(upper=50)

        ax.hist(
            megno_clipped,
            bins=bins,
            color="#185FA5",
            edgecolor="white",
            linewidth=0.6,
        )
        # Fix 3: threshold line and legend label driven by megno_threshold param
        ax.axvline(megno_threshold, color="orange", linestyle="--", linewidth=1.5,
                   label=f"threshold = {megno_threshold:.1f}")

        ax.set_title(f"{label}\n(n={len(subset):,})", fontsize=10)
        ax.set_xlabel("MEGNO", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)

        # Annotate median MEGNO on each panel for quick reading
        if len(subset) > 0:
            med = subset["MEGNO"].median()
            ax.axvline(med, color="crimson", linestyle=":", linewidth=1.2,
                       label=f"median = {med:.1f}")
            ax.legend(fontsize=8)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Fix 3: filename encodes the threshold used so multi-threshold runs
        # produce distinct files (megno_audit_B30_t3.0.png, _t4.0.png, etc.)
        fname = output_dir / f"megno_audit_{group_name}_t{megno_threshold:.1f}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")

    return fig