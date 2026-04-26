"""
ml_predictor/run_baseline.py
==================
Multi-model baseline: 5 classifiers × 6 feature groups + Macro F1 benchmarking.

Usage
-----
  # From project root:
  python -m ml_predictor.run_baseline --data data/dataset3/metadata3.csv

  # Quick smoke test on synthetic data (no real dataset needed):
  python -m ml_predictor.run_baseline --smoke-test

  # Run a single model and feature group:
  python -m ml_predictor.run_baseline --smoke-test --model lightgbm --feature-group C

  # Run full baseline then tune XGBoost on group C:
  python -m ml_predictor.run_baseline --data data/dataset3/metadata3.csv --tune-xgb

Pipeline
--------
  1. Load metadata CSV (or generate synthetic data for smoke test)
  2. Run leakage sanity check (dt, n_steps → outcome predictability)
  3. For each model × feature group:
       a. Train with stratified CV (train_model_cv)
       b. Evaluate OOF predictions (evaluate)
       c. Optional two-stage comparison (LightGBM only)
       d. Save per-group plots
  4. Save all_results.csv  — one row per (model, feature_group)
  5. Save best_model.csv   — single best row by macro F1

  If --tune-xgb is set (runs after the baseline loop):
  6. Optuna search over XGBoost hyperparameters on feature group C
  7. Train tuned model with best params (train_xgb_tuned_cv)
  8. Evaluate and save xgboost_tuned_C.csv
  9. Update best_model.csv if tuned model wins

Interpreting results
--------------------
  Macro F1 > 0.70   : Strong early-time signal — model learning real physics
  Macro F1 0.50–0.70: Moderate signal — some regimes near chaos horizon
  Macro F1 < 0.50   : Poor signal — check leakage, window definition,
                       or class imbalance within regimes

  Compare models within the same feature group to isolate model capacity.
  Compare feature groups within the same model to isolate information content.
  A model that gains nothing from w20 vs w5 features suggests the instability
  signal saturates early — consistent with rapid chaos onset at close encounters.

Notes on synthetic data (smoke test)
-------------------------------------
  The smoke test generates synthetic data approximating the real three-body
  dataset structure: four regimes with distinct class distributions, physical-
  scale features with realistic ranges. NOT physically accurate — only exists
  to verify the pipeline runs end-to-end without a real simulation dataset.
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running as `python -m ml_predictor.run_baseline` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_predictor.features  import describe_features, get_feature_matrix, get_labels
from ml_predictor.trainer   import (
    train_model_cv,
    train_two_stage_cv,
    train_xgb_tuned_cv,
    tune_xgboost_optuna,
    MODEL_REGISTRY,
)
from ml_predictor.evaluator import (
    evaluate,
    leakage_sanity_check,
    save_results_csv,
    save_best_model_csv,
)
from ml_predictor.visualiser import plot_all




# ── Synthetic data generator (smoke test only) ───────────────────────────────

def _make_synthetic_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic three-body-like dataset for pipeline testing.

    Reflects the ACTUAL feature set:
      IC:  q12, q13, q23, r12_init, r3_sep, v3_frac, v3_angle,
           epsilon_total, h_total
      DYN: dE_max_w5/w20, dL_max_w5/w20, e12/13/23_std_w5/w20,
           r_min_12/13/23_w5/w20

    Class distributions per regime match expected physics:
      hierarchical  : 70% stable,   10% unstable,  20% chaotic
      asymmetric    : 40% stable,   25% unstable,  35% chaotic
      compact_equal : 15% stable,   45% unstable,  40% chaotic
      scatter       : 25% stable,   40% unstable,  35% chaotic
    """
    rng = np.random.default_rng(seed)

    regime_specs = {
        "hierarchical":  {"n": n // 4,        "p": [0.70, 0.10, 0.20]},
        "asymmetric":    {"n": n // 4,        "p": [0.40, 0.25, 0.35]},
        "compact_equal": {"n": n // 4,        "p": [0.15, 0.45, 0.40]},
        "scatter":       {"n": n - 3*(n//4),  "p": [0.25, 0.40, 0.35]},
    }

    rows = []
    for regime, spec in regime_specs.items():
        n_reg  = spec["n"]
        labels = rng.choice([0, 1, 2], size=n_reg, p=spec["p"])

        for label in labels:
            r12 = rng.uniform(0.5, 4.0)
            r3  = r12 * rng.uniform(1.0 if label == 1 else 3.0, 8.0)

            r_min_scale  = {0: 0.8,  1: 0.15, 2: 0.4 }[label]
            e_std_scale  = {0: 0.02, 1: 0.08, 2: 0.15}[label]
            dE_scale     = {0: 1e-5, 1: 5e-4, 2: 1e-4}[label]
            dL_scale     = {0: 1e-5, 1: 5e-4, 2: 1e-4}[label]

            rows.append({
                # Meta
                "idx":    len(rows),
                "regime": regime,

                # IC features
                "epsilon_total": rng.uniform(-2.0, -0.1),
                "h_total":       rng.uniform(0.1, 5.0),
                "q12":           rng.uniform(0.1, 1.0),
                "q13":           rng.uniform(0.01, 1.0),
                "q23":           rng.uniform(0.01, 1.0),
                "r12_init":      r12,
                "r3_sep":        r3,
                "v3_frac":       rng.uniform(0.4, 1.35),
                "v3_angle":      rng.uniform(0.0, 2 * np.pi),

                # w5 dynamics
                "dE_max_w5":     abs(rng.normal(0, dE_scale)),
                "dL_max_w5":     abs(rng.normal(0, dL_scale)),
                "e12_std_w5":    abs(rng.normal(0, e_std_scale)) + 0.005,
                "e13_std_w5":    abs(rng.normal(0, e_std_scale)) + 0.005,
                "e23_std_w5":    abs(rng.normal(0, e_std_scale)) + 0.005,
                "r_min_12_w5":   max(r12 * rng.uniform(0.3, 1.0) * r_min_scale, 0.01),
                "r_min_13_w5":   max(r3  * rng.uniform(0.1, 0.5) * r_min_scale, 0.01),
                "r_min_23_w5":   max(r3  * rng.uniform(0.1, 0.5) * r_min_scale, 0.01),

                # w20 dynamics (longer window → stronger signal for chaotic)
                "dE_max_w20":    abs(rng.normal(0, dE_scale * 1.5)),
                "dL_max_w20":    abs(rng.normal(0, dL_scale * 1.5)),
                "e12_std_w20":   abs(rng.normal(0, e_std_scale * 1.5)) + 0.005,
                "e13_std_w20":   abs(rng.normal(0, e_std_scale * 1.5)) + 0.005,
                "e23_std_w20":   abs(rng.normal(0, e_std_scale * 1.5)) + 0.005,
                "r_min_12_w20":  max(r12 * rng.uniform(0.3, 1.0) * r_min_scale, 0.01),
                "r_min_13_w20":  max(r3  * rng.uniform(0.1, 0.5) * r_min_scale, 0.01),
                "r_min_23_w20":  max(r3  * rng.uniform(0.1, 0.5) * r_min_scale, 0.01),

                # Labels
                "outcome_class": label,
                "outcome":       ["stable", "unstable", "chaotic"][label],
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_baseline(
    data_path:     Path | None = None,
    output_dir:    Path        = Path("results/baseline"),
    n_folds:       int         = 5,
    two_stage:     bool        = True,
    smoke_test:    bool        = False,
    save_plots:    bool        = True,
    verbose:       bool        = True,
    feature_group: str | None  = None,
    model:         str | None  = None,
    tune_xgb:      bool        = False,
    n_trials:      int         = 75,
) -> dict:
    """
    Full multi-model baseline pipeline.

    Runs every combination of (model, feature_group) unless restricted by
    the model or feature_group arguments. Optuna tuning for XGBoost on
    feature group C is a separate post-baseline step — it does not affect
    the baseline loop in any way and never mutates MODEL_REGISTRY.

    Parameters
    ----------
    data_path     : Path to metadata CSV. Required unless smoke_test=True.
    output_dir    : Root directory for all output (plots, CSVs).
    n_folds       : Stratified CV folds (default 5).
    two_stage     : Run LightGBM two-stage comparison per feature group.
    smoke_test    : Use synthetic data instead of real dataset.
    save_plots    : Save confusion matrix, calibration, importance plots.
    verbose       : Print per-fold progress and per-run evaluation summary.
    feature_group : One of A–F. Default = run all.
    model         : One of the MODEL_REGISTRY keys. Default = run all.
    tune_xgb      : Run Optuna tuning for XGBoost on feature group C after
                    the baseline loop. Saves xgboost_tuned_C.csv separately.
    n_trials      : Number of Optuna trials (used only if tune_xgb=True).

    Returns
    -------
    dict keyed by (model_name, feature_group) → {results, report, summary}
    Also contains key ("xgboost_tuned", "C") if tune_xgb=True.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if smoke_test:
        print("── SMOKE TEST MODE ──")
        df = _make_synthetic_dataset(n=600)
    elif data_path is not None:
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Provide --data <path> or use --smoke-test")

    print(f"Dataset: {len(df)} samples", end="")
    if "regime" in df.columns:
        regime_counts = df["regime"].value_counts()
        print(f"  |  regimes: " +
              ", ".join(f"{r}={c}" for r, c in regime_counts.items()))
    else:
        print()

    label_counts = df["outcome_class"].value_counts().sort_index()
    names = ["stable", "unstable", "chaotic"]
    print("Labels: " + "  ".join(
        f"{names[int(k)]}={v} ({100*v/len(df):.0f}%)"
        for k, v in label_counts.items()
    ))

    # ── 2. Leakage check ──────────────────────────────────────────────────────
    print("\n── Leakage check ──")
    leakage_result = leakage_sanity_check(df, verbose=True)

    # describe_features is a data integrity checkpoint — run silently,
    # result available via return value but not printed to avoid clutter.
    describe_features(df, verbose=False)

    # ── 3. Resolve run scope ──────────────────────────────────────────────────
    ALL_GROUPS = ["A", "B", "C", "D", "E", "F"]
    ALL_MODELS = list(MODEL_REGISTRY.keys())

    groups_to_run = [feature_group] if feature_group is not None else ALL_GROUPS
    models_to_run = [model]         if model         is not None else ALL_MODELS

    n_runs = len(groups_to_run) * len(models_to_run)
    print(f"\n── Baseline: {len(models_to_run)} models × "
          f"{len(groups_to_run)} feature groups = {n_runs} runs ──")

    all_outputs = {}
    all_reports = []

    # ── 4. Baseline loop ──────────────────────────────────────────────────────
    for fg in groups_to_run:
        for model_name in models_to_run:

            print(f"\n  Training {model_name.upper()} | group {fg} ...")

            results = train_model_cv(
                df,
                model_name    = model_name,
                n_folds       = n_folds,
                feature_group = fg,
                verbose       = verbose,
            )

            report = evaluate(
                results,
                df            = df,
                model_name    = model_name,
                feature_group = fg,
                verbose       = verbose,
            )
            all_reports.append(report)

            # ── Two-stage (LightGBM only) ─────────────────────────────────────
            stage_results = None
            if two_stage and model_name == "lightgbm":
                print(f"  Two-stage (LightGBM | group {fg}) ...")
                s1_results, s2_results = train_two_stage_cv(
                    df,
                    n_folds       = n_folds,
                    feature_group = fg,
                    verbose       = verbose,
                )
                stage_results = (s1_results, s2_results)

                if verbose:
                    direct_ba = float(np.mean(results.fold_balanced_acc))
                    s1_ba     = float(np.mean(s1_results.fold_balanced_acc))
                    s2_ba     = float(np.mean(s2_results.fold_balanced_acc))
                    print(f"  Two-stage bal.acc — direct: {direct_ba:.4f} | "
                          f"stage1: {s1_ba:.4f} | stage2: {s2_ba:.4f}")

            # ── Plots ─────────────────────────────────────────────────────────
            if save_plots:
                group_dir = output_dir / f"group_{fg}" / model_name
                group_dir.mkdir(parents=True, exist_ok=True)
                plot_all(results=results, report=report, output_dir=group_dir)

            key = (model_name, fg)
            all_outputs[key] = {
                "results":       results,
                "report":        report,
                "stage_results": stage_results,
                "summary": {
                    "model":            model_name,
                    "feature_group":    fg,
                    "n_samples":        len(df),
                    "n_features":       len(results.feature_names),
                    "n_folds":          n_folds,
                    "oof_macro_f1":     results.mean_macro_f1,
                    "oof_macro_f1_std": results.std_macro_f1,
                    "oof_balanced_acc": float(np.mean(results.fold_balanced_acc)),
                    "oof_brier":        float(np.mean(results.fold_brier)),
                    "leakage_status":   leakage_result.get("warning_level", "unknown"),
                    "smoke_test":       smoke_test,
                },
            }

    # ── 5. Save baseline CSVs ─────────────────────────────────────────────────
    print("\n── Saving baseline results ──")
    df_all  = save_results_csv(all_reports, output_dir=output_dir)
    df_best = save_best_model_csv(df_all,   output_dir=output_dir)

    print("\n── Leaderboard (macro F1) ──")
    print(
        df_all[["model", "feature_group", "macro_f1", "macro_f1_std",
                "balanced_acc", "mean_brier"]]
        .to_string(index=False)
    )

    # ── 6. XGBoost hyperparameter tuning (post-baseline, group C only) ────────
    if tune_xgb:
        _run_xgb_tuning(
            df         = df,
            output_dir = output_dir,
            n_folds    = n_folds,
            n_trials   = n_trials,
            save_plots = save_plots,
            verbose    = verbose,
            all_outputs= all_outputs,
            df_all     = df_all,
        )

    return all_outputs


# ── XGBoost tuning step (separated for clarity) ───────────────────────────────

def _run_xgb_tuning(
    df:          pd.DataFrame,
    output_dir:  Path,
    n_folds:     int,
    n_trials:    int,
    save_plots:  bool,
    verbose:     bool,
    all_outputs: dict,
    df_all:      pd.DataFrame,
) -> None:
    """
    Post-baseline Optuna tuning for XGBoost on feature group C.

    Intentionally separated from run_baseline's main loop so it is
    impossible for tuning to affect baseline results:
      - MODEL_REGISTRY is never touched
      - best_params is a local variable passed directly to train_xgb_tuned_cv
      - the tuned result is written to xgboost_tuned_C.csv, separate from
        all_results.csv, so the two are never conflated

    best_model.csv is updated to include the tuned row only if it beats
    the current best — it is re-derived from the combined report list.
    """
    TUNING_GROUP = "C"

    print(f"\n── XGBoost Optuna tuning | group {TUNING_GROUP} | {n_trials} trials ──")

    study, best_params = tune_xgboost_optuna(
        df            = df,
        n_trials      = n_trials,
        n_folds       = n_folds,
        feature_group = TUNING_GROUP,
    )

    print(f"\n  Training XGBoost (tuned) | group {TUNING_GROUP} ...")

    tuned_results = train_xgb_tuned_cv(
        df            = df,
        best_params   = best_params,
        n_folds       = n_folds,
        feature_group = TUNING_GROUP,
        verbose       = verbose,
    )

    tuned_report = evaluate(
        tuned_results,
        df            = df,
        model_name    = "xgboost_tuned",
        feature_group = TUNING_GROUP,
        verbose       = verbose,
    )

    # ── Save tuned result to its own CSV ──────────────────────────────────────
    print("\n── Saving tuned result ──")
    df_tuned = save_results_csv(
        [tuned_report],
        output_dir = output_dir / "tuned",
    )
    # Rename the file to be explicit about what it contains
    src  = output_dir / "tuned" / "all_results.csv"
    dest = output_dir / "tuned" / "xgboost_tuned_C.csv"
    src.rename(dest)
    print(f"  Renamed → {dest}")

    # ── Re-derive best_model.csv across baseline + tuned ─────────────────────
    # Combine the baseline df_all with the single tuned row and recompute.
    # This ensures best_model.csv always reflects the globally best result.
    df_combined = pd.concat([df_all, df_tuned], ignore_index=True)
    save_best_model_csv(df_combined, output_dir=output_dir)

    # ── Optional plots for tuned model ───────────────────────────────────────
    if save_plots:
        tuned_dir = output_dir / "tuned" / "xgboost_tuned"
        tuned_dir.mkdir(parents=True, exist_ok=True)
        plot_all(results=tuned_results, report=tuned_report, output_dir=tuned_dir)

    # ── Store in all_outputs for caller inspection ────────────────────────────
    all_outputs[("xgboost_tuned", TUNING_GROUP)] = {
        "results": tuned_results,
        "report":  tuned_report,
        "study":   study,
        "best_params": best_params,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Multi-model three-body instability baseline"
    )
    p.add_argument("--data",          type=Path, default=None,
                   help="Path to metadata CSV")
    p.add_argument("--output",        type=Path, default=Path("results/baseline"),
                   help="Output directory for plots and CSVs")
    p.add_argument("--folds",         type=int,  default=5,
                   help="Number of CV folds")
    p.add_argument("--no-two-stage",  action="store_true",
                   help="Skip two-stage LightGBM classifier")
    p.add_argument("--no-plots",      action="store_true",
                   help="Skip plot generation")
    p.add_argument("--smoke-test",    action="store_true",
                   help="Run on synthetic data (no real dataset needed)")
    p.add_argument("--feature-group", type=str, default=None,
                   help="Feature group A–F. Default = run all.")
    p.add_argument("--model",         type=str, default=None,
                   help=f"Model to run. One of: {list(MODEL_REGISTRY.keys())}. "
                        "Default = run all.")
    p.add_argument("--tune-xgb",      action="store_true",
                   help="Run Optuna tuning for XGBoost on feature group C "
                        "after the baseline loop. Saves xgboost_tuned_C.csv.")
    p.add_argument("--n-trials",      type=int, default=75,
                   help="Number of Optuna trials for XGBoost tuning (default 75).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_baseline(
        data_path     = args.data,
        output_dir    = args.output,
        n_folds       = args.folds,
        two_stage     = not args.no_two_stage,
        smoke_test    = args.smoke_test,
        save_plots    = not args.no_plots,
        feature_group = args.feature_group,
        model         = args.model,
        tune_xgb      = args.tune_xgb,
        n_trials      = args.n_trials,
    )