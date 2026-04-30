"""
ml_predictor/tune_lightgbm.py
==============================

Optuna-based hyperparameter tuning for LightGBM on a chosen feature group.

Design
------
  - Self-contained tuning pipeline: the objective CV loop never calls
    train_model_cv and never touches MODEL_REGISTRY.
  - Uses Optuna's MedianPruner to kill unpromising trials early via
    per-fold intermediate reporting — saves ~30–40% of compute on a
    100-trial search.
  - Saves tuned results to {output_dir}/tuned/lightgbm_tuned_{group}.csv
    alongside an Optuna trial history CSV for post-hoc search space analysis.

Recommended usage
-----------------
  # From project root (feature group must be specified explicitly):
  python -m ml_predictor.tune_lightgbm \\
      --data data/dataset3/metadata3.csv \\
      --feature-group C+ \\
      --n-trials 100

  # Smoke test (no real data):
  python -m ml_predictor.tune_lightgbm \\
      --smoke-test \\
      --feature-group C+ \\
      --n-trials 10

Hyperparameter search space (physically motivated)
--------------------------------------------------
  num_leaves       : Controls model complexity. LightGBM grows leaf-wise;
                     too many leaves -> overfits on chaotic class (minority).
                     Range 16-128 covers shallow (interpretable) to deep.
  max_depth        : Hard depth cap to prevent individual leaves from
                     memorising rare close-encounter configurations.
  min_child_samples: Minimum samples per leaf. Critical for minority class
                     (chaotic in hierarchical regime, unstable in compact).
                     Higher values -> more conservative splits.
  learning_rate    : Log-uniform. Boosting dynamics are log-scale; small
                     rates with early stopping consistently outperform large.
  subsample        : Row subsampling per tree. Decorrelates trees on
                     regime-structured data (regime co-varies with class).
                     Requires subsample_freq > 0 to take effect in LightGBM.
  colsample_bytree : Feature subsampling. Important when IC features
                     (q12/q13/q23) are correlated -- forces trees to find
                     alternative split paths.
  reg_alpha        : L1 regularisation. Can zero out redundant mass ratios
                     (q12/q13/q23 are collinear by construction).
  reg_lambda       : L2 regularisation. General overfitting control.
  min_split_gain   : Minimum loss reduction to make a split. Suppresses
                     splits driven by numerical noise in r_min features
                     (can be very small for some trajectories).
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_predictor.features   import get_feature_matrix, get_labels
from ml_predictor.trainer    import (
    CVResults,
    sample_weights_from_labels,
    EARLY_STOPPING_ROUNDS,
    N_CLASSES,
)
from ml_predictor.evaluator  import evaluate, save_results_csv
from ml_predictor.visualiser import plot_all
from ml_predictor.run_baseline import _make_synthetic_dataset


# ─────────────────────────────────────────────────────────────────────────────
# FIXED STRUCTURAL PARAMS (shared between objective and final training)
# ─────────────────────────────────────────────────────────────────────────────

# Centralised here so the Optuna objective and train_lgbm_tuned_cv are
# guaranteed to evaluate the same model family. Previously these were
# duplicated with a key omission (subsample_freq) in the objective, meaning
# the searched model differed from the finally trained model.

_LGBM_FIXED_PARAMS: dict = {
    "objective":      "multiclass",
    "num_class":      N_CLASSES,
    "metric":         "multi_logloss",
    "n_estimators":   500,
    "subsample_freq": 1,   # REQUIRED: activates row subsampling (subsample param)
    "random_state":   42,
    "verbose":       -1,
    "n_jobs":        -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def _optuna_objective_lgbm(
    trial:         optuna.Trial,
    df:            pd.DataFrame,
    n_folds:       int,
    feature_group: str,
) -> float:
    """
    Optuna objective for LightGBM hyperparameter search.

    Uses MedianPruner via trial.report / trial.should_prune on per-fold
    intermediate F1 values. Pruning fires after the 2nd fold if the running
    mean is below the median of completed trials -- saves ~30-40% of compute
    on a 100-trial search.

    The CV loop is fully self-contained -- no MODEL_REGISTRY involvement.

    Fix notes vs original:
      - subsample_freq=1 now included via _LGBM_FIXED_PARAMS so subsample
        actually takes effect during search (was silently disabled before).
      - predict_proba + argmax used for F1 computation, consistent with
        train_lgbm_tuned_cv (was using model.predict directly).
    """
    trial_params = {
        "num_leaves":        trial.suggest_int  ("num_leaves",        16,   128           ),
        "max_depth":         trial.suggest_int  ("max_depth",         3,    8             ),
        "min_child_samples": trial.suggest_int  ("min_child_samples", 5,    100           ),
        "learning_rate":     trial.suggest_float("learning_rate",     0.01, 0.2, log=True ),
        "subsample":         trial.suggest_float("subsample",         0.6,  1.0           ),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.6,  1.0           ),
        "reg_alpha":         trial.suggest_float("reg_alpha",         1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",        1e-3, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain",    0.0,  1.0           ),
    }

    # Merge tunable params with fixed structural params -- single source of
    # truth ensures objective and final training evaluate identical model family.
    params = {**_LGBM_FIXED_PARAMS, **trial_params}

    X     = get_feature_matrix(df, feature_group=feature_group)
    y     = get_labels(df).values
    X_arr = X.values.astype(np.float32)
    feature_names = list(X.columns)

    min_count   = np.bincount(y)[np.bincount(y) > 0].min()
    safe_splits = max(2, min(n_folds, min_count))

    skf    = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
    scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y)):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y[tr_idx],     y[va_idx]
        sw          = sample_weights_from_labels(y_tr)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight = sw,
            eval_set      = [(X_va, y_va)],
            callbacks     = [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # FIX: use predict_proba + argmax, consistent with train_lgbm_tuned_cv.
        # model.predict() was used before -- diverges from the final evaluation
        # prediction path and bypasses the named-feature DataFrame interface.
        X_va_df = pd.DataFrame(X_va, columns=feature_names)
        proba   = model.predict_proba(X_va_df).astype(np.float32)
        pred    = np.argmax(proba, axis=1)

        score = f1_score(y_va, pred, average="macro", zero_division=0)
        scores.append(score)

        # Report intermediate value for pruning
        trial.report(float(np.mean(scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# TUNING ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def tune_lightgbm_optuna(
    df:            pd.DataFrame,
    feature_group: str,
    n_trials:      int = 100,
    n_folds:       int = 5,
) -> tuple[optuna.Study, dict]:
    """
    Bayesian hyperparameter search for LightGBM via Optuna.

    Uses MedianPruner -- prunes a trial if its intermediate F1 after fold k
    is below the median of all completed trials at step k. This is safe here
    because the fold ordering is deterministic (same random_state=42 split).

    Parameters
    ----------
    df            : Full dataset DataFrame.
    feature_group : Feature configuration to tune on.
    n_trials      : Optuna trials. 100 is sufficient for 9-param search;
                    diminishing returns beyond ~150.
    n_folds       : CV folds inside each trial.

    Returns
    -------
    study      : Completed Optuna study (inspect with study.trials_dataframe()).
    best_params : Ready-to-use LightGBM param dict (tuned + fixed merged).
    """
    N_REPORT_INTERVAL = 25

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials = 10,   # don't prune the first 10 trials (warmup)
        n_warmup_steps   = 2,    # don't prune before fold 2
        interval_steps   = 1,
    )

    def _progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        n          = trial.number + 1
        n_pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n % N_REPORT_INTERVAL == 0 or n == n_trials:
            print(f"  [Optuna] Trial {n:>3}/{n_trials} | "
                  f"best macro F1 = {study.best_value:.4f} | "
                  f"complete={n_complete}  pruned={n_pruned}")

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: _optuna_objective_lgbm(trial, df, n_folds, feature_group),
        n_trials          = n_trials,
        callbacks         = [_progress_callback],
        show_progress_bar = False,
    )

    # Merge best tunable params with fixed structural params.
    # _LGBM_FIXED_PARAMS is the single source of truth -- no duplication.
    best_params = {**_LGBM_FIXED_PARAMS, **study.best_params}

    n_pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n  [Optuna] Search complete -- best macro F1: {study.best_value:.4f}")
    print(f"  Trials: {n_complete} complete, {n_pruned} pruned")
    print(f"  Best params: " +
          ", ".join(
              f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
              for k, v in study.best_params.items()
          ))

    return study, best_params


# ─────────────────────────────────────────────────────────────────────────────
# FINAL TRAINING WITH TUNED PARAMS
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm_tuned_cv(
    df:            pd.DataFrame,
    best_params:   dict,
    feature_group: str,
    n_folds:       int  = 5,
    verbose:       bool = True,
) -> CVResults:
    """
    Train LightGBM with Optuna-tuned hyperparameters via stratified CV.

    Intentionally separate from train_model_cv -- accepts an explicit params
    dict, never reads from MODEL_REGISTRY. The tuned result is a clean
    additional data point, not a replacement of the baseline.

    Parameters
    ----------
    df            : Full dataset DataFrame.
    best_params   : Complete LightGBM param dict from tune_lightgbm_optuna().
    feature_group : Should match what was tuned on.
    n_folds       : CV folds.
    verbose       : Print per-fold summary line.

    Returns
    -------
    CVResults with model_name = "lightgbm_tuned".
    """
    X       = get_feature_matrix(df, feature_group=feature_group)
    y       = get_labels(df).values
    regimes = df["regime"].values if "regime" in df.columns else np.zeros(len(df))

    feature_names = list(X.columns)
    X_arr         = X.values.astype(np.float32)

    min_count   = np.bincount(y)[np.bincount(y) > 0].min()
    safe_splits = max(2, min(n_folds, min_count))

    skf     = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
    results = CVResults(model_name="lightgbm_tuned", feature_names=feature_names)

    oof_true  = np.full(len(y), -1, dtype=int)
    oof_pred  = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y)):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y[tr_idx],     y[va_idx]
        sw          = sample_weights_from_labels(y_tr)

        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_tr, y_tr,
            sample_weight = sw,
            eval_set      = [(X_va, y_va)],
            callbacks     = [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        X_va_df   = pd.DataFrame(X_va, columns=feature_names)
        proba_val = model.predict_proba(X_va_df).astype(np.float32)
        pred_val  = np.argmax(proba_val, axis=1)

        oof_true[va_idx]  = y_va
        oof_pred[va_idx]  = pred_val
        oof_proba[va_idx] = proba_val

        macro_f1 = f1_score(y_va, pred_val, average="macro", zero_division=0)
        bal_acc  = balanced_accuracy_score(y_va, pred_val)
        cm       = confusion_matrix(y_va, pred_val, labels=[0, 1, 2])
        brier    = float(np.mean([
            brier_score_loss((y_va == cls).astype(float), proba_val[:, cls])
            for cls in range(N_CLASSES)
        ]))

        results.fold_macro_f1.append(macro_f1)
        results.fold_balanced_acc.append(bal_acc)
        results.fold_brier.append(brier)   # FIX: was missing -- caused fold_brier=[] -> mean_brier=nan in CSV
        results.fold_cms.append(cm)
        results.models.append(model)

        if verbose:
            print(f"  Fold {fold_idx+1}/{safe_splits} | "
                  f"Macro F1={macro_f1:.4f} | "
                  f"Bal.Acc={bal_acc:.4f} | "
                  f"Brier={brier:.4f}")

    results.oof_true   = oof_true
    results.oof_pred   = oof_pred
    results.oof_proba  = oof_proba
    results.oof_regime = regimes

    if verbose:
        print(f"  OOF Macro F1 : {results.mean_macro_f1:.4f} +/- {results.std_macro_f1:.4f}")
        print(f"  OOF Bal. Acc : {np.mean(results.fold_balanced_acc):.4f}")
        print(f"  OOF Brier    : {np.mean(results.fold_brier):.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_lightgbm_tuning(
    data_path:     Path | None,
    feature_group: str,
    output_dir:    Path = Path("results/baseline"),
    n_trials:      int  = 100,
    n_folds:       int  = 5,
    smoke_test:    bool = False,
    save_plots:    bool = True,
    verbose:       bool = True,
) -> dict:
    """
    Full LightGBM tuning pipeline.

      1. Load data (or generate synthetic for smoke test)
      2. Run Optuna search (n_trials, with MedianPruner)
      3. Train final model with best params (full CV)
      4. Evaluate and save lightgbm_tuned_{feature_group}.csv
      5. Save Optuna trial history CSV
      6. Optionally save plots

    Parameters
    ----------
    data_path     : Path to metadata CSV. Required unless smoke_test=True.
    feature_group : Feature configuration to tune on. Must be specified.
    output_dir    : Root results directory. Tuned outputs go to
                    {output_dir}/tuned/lightgbm_tuned_{feature_group}.csv.
    n_trials      : Optuna trials.
    n_folds       : CV folds.
    smoke_test    : Use synthetic data.
    save_plots    : Save confusion matrix, calibration, importance plots.
    verbose       : Print per-fold progress.

    Returns
    -------
    dict with keys: study, best_params, results, report
    """
    output_dir = Path(output_dir)
    tuned_dir  = output_dir / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if smoke_test:
        print("── SMOKE TEST MODE ──")
        df = _make_synthetic_dataset(n=600)
    elif data_path is not None:
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Provide --data <path> or use --smoke-test")

    print(f"Dataset: {len(df)} samples | Feature group: {feature_group}")
    label_counts = df["outcome_class"].value_counts().sort_index()
    names = ["stable", "unstable", "chaotic"]
    print("Labels: " + "  ".join(
        f"{names[int(k)]}={v} ({100*v/len(df):.0f}%)"
        for k, v in label_counts.items()
    ))

    # ── Optuna search ─────────────────────────────────────────────────────────
    print(f"\n── LightGBM Optuna tuning | group {feature_group} | {n_trials} trials ──")
    study, best_params = tune_lightgbm_optuna(
        df            = df,
        feature_group = feature_group,
        n_trials      = n_trials,
        n_folds       = n_folds,
    )

    # ── Train final model ─────────────────────────────────────────────────────
    print(f"\n── Training LightGBM (tuned) | group {feature_group} ──")
    tuned_results = train_lgbm_tuned_cv(
        df            = df,
        best_params   = best_params,
        feature_group = feature_group,
        n_folds       = n_folds,
        verbose       = verbose,
    )

    tuned_report = evaluate(
        tuned_results,
        df            = df,
        model_name    = "lightgbm_tuned",
        feature_group = feature_group,
        verbose       = True,
    )

    # ── Save tuned CSV ────────────────────────────────────────────────────────
    print("\n── Saving tuned result ──")
    save_results_csv([tuned_report], output_dir=tuned_dir)

    src  = tuned_dir / "all_results.csv"
    dest = tuned_dir / f"lightgbm_tuned_{feature_group}.csv"

    # FIX: use Path.replace() instead of Path.rename().
    # rename() raises FileExistsError on Windows if dest exists (re-runs fail).
    # replace() is atomic on POSIX and overwrites safely on all platforms.
    if src.exists():
        src.replace(dest)
        print(f"  Renamed -> {dest}")

    # ── Save Optuna trial history ─────────────────────────────────────────────
    trials_path = tuned_dir / f"optuna_trials_{feature_group}.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    print(f"  Saved trial history -> {trials_path}")

    # ── Optional plots ────────────────────────────────────────────────────────
    if save_plots:
        plot_dir = tuned_dir / f"lightgbm_tuned_{feature_group}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_all(results=tuned_results, report=tuned_report, output_dir=plot_dir)

    return {
        "study":       study,
        "best_params": best_params,
        "results":     tuned_results,
        "report":      tuned_report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="LightGBM Optuna tuning for three-body instability"
    )
    p.add_argument("--data",          type=Path, default=None,
                   help="Path to metadata CSV")
    p.add_argument("--output",        type=Path, default=Path("results/baseline"),
                   help="Root output directory (tuned CSV written to "
                        "{output}/tuned/lightgbm_tuned_{feature_group}.csv)")
    p.add_argument("--feature-group", type=str,  required=True,
                   help="Feature group to tune on (e.g. C+, F). Required.")
    p.add_argument("--n-trials",      type=int,  default=100,
                   help="Number of Optuna trials (default: 100)")
    p.add_argument("--folds",         type=int,  default=5,
                   help="CV folds (default: 5)")
    p.add_argument("--smoke-test",    action="store_true",
                   help="Use synthetic data (no real dataset needed)")
    p.add_argument("--no-plots",      action="store_true",
                   help="Skip plot generation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_lightgbm_tuning(
        data_path     = args.data,
        feature_group = args.feature_group,
        output_dir    = args.output,
        n_trials      = args.n_trials,
        n_folds       = args.folds,
        smoke_test    = args.smoke_test,
        save_plots    = not args.no_plots,
    )