"""
ml_predictor/trainer.py
=============
Multi-model training pipeline for three-body instability classification.

Models
------
Five tree-ensemble / decision-tree classifiers are benchmarked:

  1. DecisionTree   — single tree; interpretable threshold structure.
                      Useful baseline: exposes raw decision boundaries
                      (hill_ratio < X, r_min < Y) without ensemble averaging.
                      No early stopping; depth-limited by max_depth.

  2. RandomForest   — bagged trees; reduces variance vs. single tree.
                      Particularly useful when instability signal is distributed
                      across multiple weak features (e.g. eccentricity drift +
                      angular momentum violation jointly).

  3. XGBoost        — gradient-boosted trees with L1/L2 regularisation.
                      Early stopping on val logloss. Strong on structured
                      tabular data with threshold nonlinearities.

  4. LightGBM       — histogram-based gradient boosting; fast on high-cardinality
                      continuous features (r_min_ij, eccentricity). Leaf-wise
                      growth captures asymmetric instability regions efficiently.

  5. CatBoost       — symmetric tree boosting with built-in ordered boosting.
                      Robust to correlated features (q12/q13/q23 mass ratios
                      are correlated by construction).

Design
------
  1. MODEL_REGISTRY: single source of truth for all model configs.
     Adding a 6th model = one dict entry.

  2. CVResults is model-agnostic — stores OOF arrays regardless of model type.
     model_name field added for downstream traceability.

  3. Early stopping:
       LightGBM  → lgb.early_stopping callback
       XGBoost   → early_stopping_rounds in .fit()
       CatBoost  → early_stopping_rounds param
       DT / RF   → no early stopping (train to completion)

  4. Class weighting:
       All models receive inverse-frequency sample weights per training fold.
       Weights computed from training fold ONLY — no leakage from val fold.

  5. train_model_cv() is the unified entry point.
     train_lightgbm_cv() is preserved as a backward-compatible alias.

  6. train_two_stage_cv() uses LightGBM only — it is a physically-motivated
     hierarchical classifier (Hill boundary → energy exchange boundary),
     not a benchmarking target. Running all 5 models through two-stage
     would be redundant and computationally wasteful.

Two-stage option
----------------
Stage 1: Stable vs. Not-Stable (binary, LightGBM)
Stage 2: Unstable vs. Chaotic (binary, LightGBM — on not-stable subset only)

Physical motivation:
  The stable/not-stable boundary is a geometric question (Hill criterion:
  orbital separations, mass ratios). The unstable/chaotic boundary is a
  dynamical question (energy exchange during close encounters). These are
  qualitatively different physical questions — the same feature set may
  not be optimal for both, and conflating them in a single 3-class model
  forces the model to learn two incompatible decision rules simultaneously.
"""

from __future__ import annotations
import optuna

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from ml_predictor.features import get_feature_matrix, get_labels

# ── Label constants ───────────────────────────────────────────────────────────

CLASS_NAMES = {0: "stable", 1: "unstable", 2: "chaotic"}
N_CLASSES   = 3

EARLY_STOPPING_ROUNDS = 50


# ── Model registry ────────────────────────────────────────────────────────────
#
# Each entry: model_name → dict with keys:
#   cls           : sklearn-compatible classifier class
#   params        : constructor kwargs
#   early_stop    : whether this model supports early stopping on eval set
#   fit_kwargs    : additional kwargs passed to .fit() (excluding sample_weight)
#
# Convention:
#   - All params use random_state=42 for reproducibility.
#   - Depth kept conservative (max_depth=5) across models for fair comparison
#     on small datasets (N~500–5000). Deeper trees mostly add noise given the
#     strong main effects of hill_ratio and r_min_ij.
#   - n_estimators=500 for boosting models; early stopping handles actual count.


MODEL_REGISTRY: dict[str, dict[str, Any]] = {

    "decision_tree": {
        "cls": DecisionTreeClassifier,
        "params": {
            "max_depth":        5,
            "min_samples_leaf": 10,    # prevents splits on < 10 samples
            "class_weight":     "balanced",  # DT uses class_weight, not sample_weight
            "random_state":     42,
        },
        "early_stop":  False,
        "fit_kwargs":  {},
        # Note: DecisionTreeClassifier uses class_weight="balanced" natively.
        # sample_weight is still passed for consistency but class_weight takes precedence
        # in determining split criterion weights.
    },

    "random_forest": {
        "cls": RandomForestClassifier,
        "params": {
            "n_estimators":     300,
            "max_depth":        5,
            "min_samples_leaf": 10,
            "max_features":     "sqrt",   # standard for classification
            "n_jobs":           -1,
            "random_state":     42,
        },
        "early_stop":  False,
        "fit_kwargs":  {},
        # Bagging reduces variance from single-tree threshold instability.
        # max_features="sqrt" ensures each tree sees a random feature subset,
        # decorrelating trees — critical when q12/q13/q23 are correlated.
    },

    "xgboost": {
        "cls": xgb.XGBClassifier,
        "params": {
            "objective":             "multi:softprob",
            "num_class":             N_CLASSES,
            "eval_metric":           "mlogloss",
            "n_estimators":          500,
            "max_depth":             5,
            "learning_rate":         0.05,
            "subsample":             0.8,
            "colsample_bytree":      0.8,
            "reg_alpha":             0.1,
            "reg_lambda":            1.0,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,  # XGBoost 3.x: constructor param
            "random_state":          42,
            "verbosity":             0,
            "n_jobs":                -1,
        },
        "early_stop":  True,
        "fit_kwargs":  {},
        # XGBoost 3.x moved early_stopping_rounds to the constructor.
        # eval_set is still passed in .fit() — the constructor param activates
        # early stopping only when an eval_set is provided at fit time.
    },

    "lightgbm": {
        "cls": lgb.LGBMClassifier,
        "params": {
            "objective":        "multiclass",
            "num_class":        N_CLASSES,
            "metric":           "multi_logloss",
            "num_leaves":       31,
            "max_depth":        5,
            "min_child_samples":10,
            "learning_rate":    0.05,
            "n_estimators":     500,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "random_state":     42,
            "verbose":         -1,
            "n_jobs":          -1,
        },
        "early_stop":  True,
        "fit_kwargs":  {},
    },

    "catboost": {
        "cls": CatBoostClassifier,
        "params": {
            "loss_function":        "MultiClass",
            "eval_metric":          "TotalF1",
            "iterations":           500,
            "depth":                5,
            "learning_rate":        0.05,
            "l2_leaf_reg":          1.0,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "random_seed":          42,
            "verbose":              0,
            "thread_count":         -1,
        },
        "early_stop":  True,   # handled internally via early_stopping_rounds param
        "fit_kwargs":  {},
        # CatBoost uses symmetric (oblivious) trees — each level applies the
        # same split across all leaves. This reduces overfitting on small datasets
        # and is robust to correlated features (mass ratios q12/q13/q23).
        # ordered boosting (default) prevents target leakage in training.
    },
}


# ── Data class for CV results ─────────────────────────────────────────────────

@dataclass
class CVResults:
    """
    Stores per-fold and aggregate OOF results from cross-validation.
    Model-agnostic — works for any entry in MODEL_REGISTRY.
    """

    model_name:             str            = ""

    # Per-fold scalar metrics
    fold_macro_f1:          list[float]    = field(default_factory=list)
    fold_balanced_acc:      list[float]    = field(default_factory=list)
    fold_brier:             list[float]    = field(default_factory=list)

    # Per-fold confusion matrices (N_CLASSES × N_CLASSES)
    fold_cms:               list[np.ndarray] = field(default_factory=list)

    # Out-of-fold predictions (assembled across all folds)
    oof_true:               np.ndarray    = field(default_factory=lambda: np.array([]))
    oof_pred:               np.ndarray    = field(default_factory=lambda: np.array([]))
    oof_proba:              np.ndarray    = field(default_factory=lambda: np.array([]))
    oof_regime:             np.ndarray    = field(default_factory=lambda: np.array([]))

    # Trained models (one per fold) and feature metadata
    models:                 list          = field(default_factory=list)
    feature_names:          list[str]     = field(default_factory=list)

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean(self.fold_macro_f1))

    @property
    def std_macro_f1(self) -> float:
        return float(np.std(self.fold_macro_f1))

    @property
    def aggregate_cm(self) -> np.ndarray:
        """Sum of per-fold confusion matrices — full OOF performance."""
        return sum(self.fold_cms)

    # Aliases for evaluator.py compatibility
    @property
    def y_true(self):
        return self.oof_true

    @property
    def y_pred(self):
        return self.oof_pred

    @property
    def y_proba(self):
        return self.oof_proba


# ── Class weight utilities ────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute inverse-frequency class weights from a label array.

    In three-body datasets, class distributions are strongly regime-dependent
    (hierarchical → stable-dominated; compact_equal → chaotic-dominated).
    Inverse-frequency weighting corrects for this without requiring regime
    labels, ensuring the model pays equal attention to all three outcome types.

    Weights computed from training fold ONLY — computing from the full dataset
    would leak the validation fold's class distribution into training.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    weights = {}
    for cls, cnt in zip(classes, counts):
        weights[int(cls)] = n_total / (N_CLASSES * cnt)
    return weights


def sample_weights_from_labels(y: np.ndarray) -> np.ndarray:
    """Convert class weight dict to per-sample weight array."""
    cw = compute_class_weights(y)
    return np.array([cw[int(yi)] for yi in y])


# ── Core unified training function ────────────────────────────────────────────

def train_model_cv(
    df:            pd.DataFrame,
    model_name:    str  = "lightgbm",
    n_folds:       int  = 5,
    feature_group: str  = "C",
    verbose:       bool = True,
) -> CVResults:
    """
    Train any model from MODEL_REGISTRY with stratified cross-validation.

    CV loop:
      1. Stratify splits by outcome_class (preserves class balance per fold).
      2. Compute inverse-frequency sample weights from training fold only.
      3. Train with early stopping where supported (LightGBM, XGBoost, CatBoost).
      4. Assemble OOF predictions and compute macro F1, balanced accuracy, Brier.

    Parameters
    ----------
    df            : Full metadata DataFrame.
    model_name    : Key in MODEL_REGISTRY. One of:
                    'decision_tree', 'random_forest', 'xgboost',
                    'lightgbm', 'catboost'.
    n_folds       : CV folds. Use 5 (standard) or 3 for tiny datasets.
    feature_group : Feature configuration A–F (see features.py).
    verbose       : Print per-fold progress.

    Returns
    -------
    CVResults with OOF predictions, per-fold metrics, and trained models.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[model_name]

    # ── Feature matrix and labels ─────────────────────────────────────────────
    X = get_feature_matrix(df, feature_group=feature_group)
    y = get_labels(df).values
    regimes = df["regime"].values if "regime" in df.columns else np.zeros(len(df))

    feature_names = list(X.columns)
    X_arr = X.values.astype(np.float32)

    # ── Safe fold count (guard against tiny minority classes) ─────────────────
    class_counts = np.bincount(y)
    min_count    = class_counts[class_counts > 0].min()
    safe_splits  = max(2, min(n_folds, min_count))

    if verbose:
        print(f"  [{model_name.upper()} | group {feature_group}] "
              f"{len(X)} samples, {len(feature_names)} features, {safe_splits} folds")

    skf = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
    results = CVResults(model_name=model_name, feature_names=feature_names)

    # Pre-allocate OOF arrays
    oof_true  = np.full(len(y), -1, dtype=int)
    oof_pred  = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)

    # ── CV loop ───────────────────────────────────────────────────────────────
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y[train_idx],     y[val_idx]

        sw_train = sample_weights_from_labels(y_train)

        model = cfg["cls"](**cfg["params"])
        model = _fit_model(
            model, model_name, cfg,
            X_train, y_train, X_val, y_val, sw_train,
        )

        proba_val = _predict_proba(model, model_name, X_val, feature_names)
        pred_val  = np.argmax(proba_val, axis=1)

        oof_true[val_idx]  = y_val
        oof_pred[val_idx]  = pred_val
        oof_proba[val_idx] = proba_val

        # ── Per-fold metrics ──────────────────────────────────────────────────
        macro_f1 = f1_score(y_val, pred_val, average="macro", zero_division=0)
        bal_acc  = balanced_accuracy_score(y_val, pred_val)
        cm       = confusion_matrix(y_val, pred_val, labels=[0, 1, 2])

        brier_vals = []
        for cls in range(N_CLASSES):
            y_bin = (y_val == cls).astype(float)
            brier_vals.append(brier_score_loss(y_bin, proba_val[:, cls]))
        brier = float(np.mean(brier_vals))

        results.fold_macro_f1.append(macro_f1)
        results.fold_balanced_acc.append(bal_acc)
        results.fold_brier.append(brier)
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
        print(f"  OOF Macro F1 : {results.mean_macro_f1:.4f} ± {results.std_macro_f1:.4f} | "
              f"Bal.Acc : {np.mean(results.fold_balanced_acc):.4f} | "
              f"Brier : {np.mean(results.fold_brier):.4f}")

    return results


# ── Model fit / predict helpers ───────────────────────────────────────────────

def _fit_model(
    model,
    model_name: str,
    cfg:        dict,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    sw_train:   np.ndarray,
) -> Any:
    """
    Dispatch .fit() call with model-specific early stopping and weight handling.

    DecisionTree / RandomForest:
      - No eval set. Pass sample_weight directly.

    LightGBM:
      - Early stopping via lgb.early_stopping callback on val logloss.

    XGBoost:
      - Early stopping via early_stopping_rounds + eval_set in .fit().

    CatBoost:
      - early_stopping_rounds is a constructor param (already set in registry).
        Pass eval_set as Pool to .fit().
    """
    if model_name in ("decision_tree", "random_forest"):
        model.fit(X_train, y_train, sample_weight=sw_train)

    elif model_name == "lightgbm":
        model.fit(
            X_train, y_train,
            sample_weight = sw_train,
            eval_set      = [(X_val, y_val)],
            callbacks     = [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

    elif model_name == "xgboost":
        # XGBoost 3.x: early_stopping_rounds lives in the constructor.
        # Passing eval_set here activates it; verbose=False suppresses iteration log.
        model.fit(
            X_train, y_train,
            sample_weight = sw_train,
            eval_set      = [(X_val, y_val)],
            verbose       = False,
        )

    elif model_name == "catboost":
        from catboost import Pool
        train_pool = Pool(X_train, y_train, weight=sw_train)
        val_pool   = Pool(X_val,   y_val)
        model.fit(train_pool, eval_set=val_pool, verbose=False)

    return model


def _predict_proba(model, model_name: str, X: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
    """
    Uniform predict_proba interface across all 5 models.
    Returns (n_samples, N_CLASSES) float32 arrays.

    LightGBM is fitted with named features; passing a named DataFrame at
    predict time suppresses the feature-name mismatch UserWarning.
    """
    if model_name == "lightgbm" and feature_names is not None:
        X_in = pd.DataFrame(X, columns=feature_names)
    else:
        X_in = X
    proba = model.predict_proba(X_in)
    return proba.astype(np.float32)

def _optuna_objective_xgb(trial, df, n_folds, feature_group):
    """
    Optuna objective for XGBoost hyperparameter search.

    Builds the model directly from trial params — never touches MODEL_REGISTRY.
    This avoids the global mutation + restore anti-pattern, which corrupts the
    registry permanently if a trial raises an exception before restoration.

    The CV loop is inlined (not delegated to train_model_cv) so that the
    params dict is owned entirely by this function and is not shared state.
    """
    # ── Searchable hyperparameters ────────────────────────────────────────────
    # Ranges are physically motivated:
    #   max_depth 3–8   : shallow enough to avoid overfitting on N~5000,
    #                     deep enough to capture hill_ratio × r_min interactions.
    #   learning_rate   : log-uniform; boosting benefits from log-scale exploration.
    #   subsample/col   : row and column subsampling — standard regularisation.
    #   reg_alpha/lambda: L1 + L2; important when IC features (q12/q13/q23)
    #                     are correlated — L1 can zero out redundant mass ratios.
    #   min_child_weight: minimum sum of instance weights in a leaf child.
    #                     Higher values → more conservative splits on rare classes.

    trial_params = {
        "max_depth":         trial.suggest_int  ("max_depth",         3,    8          ),
        "learning_rate":     trial.suggest_float("learning_rate",     0.01, 0.2,  log=True),
        "subsample":         trial.suggest_float("subsample",         0.6,  1.0        ),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.6,  1.0        ),
        "reg_alpha":         trial.suggest_float("reg_alpha",         1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",        1e-3, 10.0, log=True),
        "min_child_weight":  trial.suggest_float("min_child_weight",  1e-2, 10.0, log=True),
    }

    # ── Fixed params (not tuned — structurally required) ─────────────────────
    params = {
        **trial_params,
        "n_estimators":          500,
        "objective":             "multi:softprob",
        "num_class":             N_CLASSES,
        "eval_metric":           "mlogloss",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state":          42,
        "verbosity":             0,
        "n_jobs":                -1,
    }

    # ── CV with fresh model instances — no registry involvement ───────────────
    X = get_feature_matrix(df, feature_group=feature_group)
    y = get_labels(df).values
    X_arr = X.values.astype(np.float32)

    class_counts = np.bincount(y)
    min_count    = class_counts[class_counts > 0].min()
    safe_splits  = max(2, min(n_folds, min_count))

    skf    = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_arr, y):
        X_tr, X_v = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_v = y[train_idx],     y[val_idx]
        sw         = sample_weights_from_labels(y_tr)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight = sw,
            eval_set      = [(X_v, y_v)],
            verbose       = False,
        )

        pred  = model.predict(X_v)
        score = f1_score(y_v, pred, average="macro", zero_division=0)
        scores.append(score)

    return float(np.mean(scores))

def tune_xgboost_optuna(
    df:            pd.DataFrame,
    n_trials:      int = 75,
    n_folds:       int = 5,
    feature_group: str = "C",
) -> tuple[optuna.Study, dict]:
    """
    Bayesian hyperparameter search for XGBoost via Optuna.

    Restricted to feature_group="C" (IC + w20) — the configuration where
    XGBoost is already the best baseline. Tuning other groups is redundant
    unless the ablation study reveals a different bottleneck.

    Design decisions:
      - Optuna stdout suppressed entirely; a single progress line per
        N_REPORT_INTERVAL trials is printed instead.
      - The study and best_params are both returned so the caller can
        (a) inspect the full trial history and (b) pass best_params
        directly to train_xgb_tuned_cv without touching MODEL_REGISTRY.

    Parameters
    ----------
    df            : Full dataset DataFrame.
    n_trials      : Number of Optuna trials (default 75 — sufficient for
                    7-param search; diminishing returns beyond ~100).
    n_folds       : CV folds inside each trial evaluation.
    feature_group : Feature configuration to tune on (should be "C").

    Returns
    -------
    study      : Completed Optuna study (full trial history).
    best_params : Ready-to-use XGBoost param dict (tuned + fixed params merged).
    """
    N_REPORT_INTERVAL = 25   # print progress every N trials

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        n = trial.number + 1
        if n % N_REPORT_INTERVAL == 0 or n == n_trials:
            print(f"  [Optuna] Trial {n:>3}/{n_trials} | "
                  f"best macro F1 = {study.best_value:.4f}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _optuna_objective_xgb(trial, df, n_folds, feature_group),
        n_trials   = n_trials,
        callbacks  = [_progress_callback],
        show_progress_bar = False,
    )

    # Merge tuned params with fixed structural params into one ready-to-use dict
    best_params = {
        **study.best_params,
        "n_estimators":          500,
        "objective":             "multi:softprob",
        "num_class":             N_CLASSES,
        "eval_metric":           "mlogloss",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state":          42,
        "verbosity":             0,
        "n_jobs":                -1,
    }

    print(f"\n  [Optuna] Search complete — best macro F1: {study.best_value:.4f}")
    print(f"  Best params: " +
          ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in study.best_params.items()))

    return study, best_params

def train_xgb_tuned_cv(
    df:            pd.DataFrame,
    best_params:   dict,
    n_folds:       int  = 5,
    feature_group: str  = "C",
    verbose:       bool = True,
) -> CVResults:
    """
    Train XGBoost with Optuna-tuned hyperparameters via stratified CV.

    This is intentionally separate from train_model_cv — it accepts an
    explicit params dict rather than reading from MODEL_REGISTRY, so the
    baseline registry is never mutated and the tuned result is a clean
    additional data point rather than a replacement.

    Physical context:
      Feature group C (IC + w20) is the sweet spot for XGBoost because:
        - IC features (hill_ratio, r_separation_ratio) provide the geometric
          stability boundary — a hard threshold the model can learn exactly.
        - w20 dynamics (dE_max, r_min_ij, e_std) carry the chaos precursor
          signal accumulated over the first 20% of the integration window.
      Tuning matters here because the optimal regularisation (reg_alpha/lambda)
      depends on the correlation structure of IC + w20 features jointly.

    Parameters
    ----------
    df            : Full dataset DataFrame.
    best_params   : Complete XGBoost param dict from tune_xgboost_optuna().
                    Must include both tuned and fixed structural params.
    n_folds       : CV folds.
    feature_group : Feature configuration — should match what was tuned on.
    verbose       : Print per-fold summary line.

    Returns
    -------
    CVResults — same structure as train_model_cv output, with
                model_name = "xgboost_tuned".
    """
    X = get_feature_matrix(df, feature_group=feature_group)
    y = get_labels(df).values
    regimes = df["regime"].values if "regime" in df.columns else np.zeros(len(df))

    feature_names = list(X.columns)
    X_arr         = X.values.astype(np.float32)

    class_counts = np.bincount(y)
    min_count    = class_counts[class_counts > 0].min()
    safe_splits  = max(2, min(n_folds, min_count))

    skf     = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
    results = CVResults(model_name="xgboost_tuned", feature_names=feature_names)

    oof_true  = np.full(len(y), -1, dtype=int)
    oof_pred  = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y[train_idx],     y[val_idx]
        sw_train        = sample_weights_from_labels(y_train)

        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train, y_train,
            sample_weight = sw_train,
            eval_set      = [(X_val, y_val)],
            verbose       = False,
        )

        proba_val = model.predict_proba(X_val).astype(np.float32)
        pred_val  = np.argmax(proba_val, axis=1)

        oof_true[val_idx]  = y_val
        oof_pred[val_idx]  = pred_val
        oof_proba[val_idx] = proba_val

        macro_f1 = f1_score(y_val, pred_val, average="macro", zero_division=0)
        bal_acc  = balanced_accuracy_score(y_val, pred_val)
        cm       = confusion_matrix(y_val, pred_val, labels=[0, 1, 2])

        brier_vals = [
            brier_score_loss((y_val == cls).astype(float), proba_val[:, cls])
            for cls in range(N_CLASSES)
        ]
        brier = float(np.mean(brier_vals))

        results.fold_macro_f1.append(macro_f1)
        results.fold_balanced_acc.append(bal_acc)
        results.fold_brier.append(brier)
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
        print(f"  OOF Macro F1 : {results.mean_macro_f1:.4f} ± {results.std_macro_f1:.4f}")
        print(f"  OOF Bal. Acc : {np.mean(results.fold_balanced_acc):.4f}")
        print(f"  OOF Brier    : {np.mean(results.fold_brier):.4f}")

    return results




def train_lightgbm_cv(
    df:            pd.DataFrame,
    n_folds:       int  = 5,
    feature_group: str  = "C",
    lgbm_params:   dict | None = None,
    verbose:       bool = True,
) -> CVResults:
    """
    Backward-compatible alias for train_model_cv(model_name='lightgbm').

    lgbm_params overrides are NOT forwarded — use train_model_cv() directly
    for custom hyperparameter control.
    """
    return train_model_cv(
        df            = df,
        model_name    = "lightgbm",
        n_folds       = n_folds,
        feature_group = feature_group,
        verbose       = verbose,
    )


# ── Two-stage classifier (LightGBM only) ─────────────────────────────────────

def train_two_stage_cv(
    df:            pd.DataFrame,
    n_folds:       int  = 5,
    feature_group: str  = "C",
    lgbm_params:   dict | None = None,
    verbose:       bool = True,
) -> tuple[CVResults, CVResults]:
    """
    Two-stage classification mirroring the physical instability hierarchy.

    Stage 1: Stable (0) vs. Not-Stable (1 or 2)   — Hill criterion boundary
    Stage 2: Unstable (1) vs. Chaotic (2)           — energy exchange boundary
             trained ONLY on not-stable samples

    Uses LightGBM exclusively — this is a physically-motivated hierarchical
    model, not a benchmarking target. The two boundaries correspond to
    qualitatively different physical processes and are intentionally separated.

    Returns
    -------
    (stage1_results, stage2_results) : CVResults for each stage.
    """
    lgbm_cfg = MODEL_REGISTRY["lightgbm"]
    base_params = {**lgbm_cfg["params"]}

    binary_params = {
        **base_params,
        "num_class": 1,
        "objective": "binary",
        "metric":    "binary_logloss",
    }

    X = get_feature_matrix(df, feature_group=feature_group)
    y = get_labels(df).values
    regimes = df["regime"].values if "regime" in df.columns else np.zeros(len(df))
    X_arr = X.values.astype(np.float32)

    # ── Stage 1: Stable vs Not-Stable ─────────────────────────────────────────
    if verbose:
        print("\n── Stage 1: Stable vs Not-Stable (LightGBM) ──")

    y_s1  = (y > 0).astype(int)
    skf1  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results1 = CVResults(model_name="lightgbm_stage1", feature_names=list(X.columns))
    oof_t1   = np.zeros(len(y), dtype=int)
    oof_p1   = np.zeros(len(y), dtype=int)
    oof_pb1  = np.zeros((len(y), 2), dtype=np.float32)

    for fold_idx, (tr, va) in enumerate(skf1.split(X_arr, y_s1)):
        X_tr, X_v = X_arr[tr], X_arr[va]
        y_tr, y_v = y_s1[tr], y_s1[va]
        sw = sample_weights_from_labels(y_tr)

        m = lgb.LGBMClassifier(**binary_params)
        m.fit(
            X_tr, y_tr, sample_weight=sw,
            eval_set  = [(X_v, y_v)],
            callbacks = [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        p  = m.predict_proba(X_v).astype(np.float32)
        yp = np.argmax(p, axis=1)

        oof_t1[va]  = y_v
        oof_p1[va]  = yp
        oof_pb1[va] = p

        f1 = f1_score(y_v, yp, average="macro", zero_division=0)
        ba = balanced_accuracy_score(y_v, yp)
        cm = confusion_matrix(y_v, yp, labels=[0, 1])

        results1.fold_macro_f1.append(f1)
        results1.fold_balanced_acc.append(ba)
        results1.fold_cms.append(cm)
        results1.models.append(m)

        if verbose:
            print(f"  Fold {fold_idx+1} | F1={f1:.4f} | Bal.Acc={ba:.4f}")

    results1.oof_true   = oof_t1
    results1.oof_pred   = oof_p1
    results1.oof_proba  = oof_pb1
    results1.oof_regime = regimes

    # ── Stage 2: Unstable vs Chaotic (not-stable subset only) ─────────────────
    if verbose:
        print("\n── Stage 2: Unstable vs Chaotic (LightGBM) ──")

    mask_ns  = y > 0
    X_ns     = X_arr[mask_ns]
    y_ns2    = (y[mask_ns] == 2).astype(int)   # 0=unstable, 1=chaotic
    reg_ns   = regimes[mask_ns]

    safe_s2  = max(2, min(n_folds, np.bincount(y_ns2).min()))
    skf2     = StratifiedKFold(n_splits=safe_s2, shuffle=True, random_state=42)

    results2 = CVResults(model_name="lightgbm_stage2", feature_names=list(X.columns))
    oof_t2   = np.zeros(len(y_ns2), dtype=int)
    oof_p2   = np.zeros(len(y_ns2), dtype=int)
    oof_pb2  = np.zeros((len(y_ns2), 2), dtype=np.float32)

    for fold_idx, (tr, va) in enumerate(skf2.split(X_ns, y_ns2)):
        X_tr, X_v = X_ns[tr], X_ns[va]
        y_tr, y_v = y_ns2[tr], y_ns2[va]
        sw = sample_weights_from_labels(y_tr)

        m = lgb.LGBMClassifier(**binary_params)
        m.fit(
            X_tr, y_tr, sample_weight=sw,
            eval_set  = [(X_v, y_v)],
            callbacks = [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        p  = m.predict_proba(X_v).astype(np.float32)
        yp = np.argmax(p, axis=1)

        oof_t2[va]  = y_v
        oof_p2[va]  = yp
        oof_pb2[va] = p

        f1 = f1_score(y_v, yp, average="macro", zero_division=0)
        ba = balanced_accuracy_score(y_v, yp)
        cm = confusion_matrix(y_v, yp, labels=[0, 1])

        results2.fold_macro_f1.append(f1)
        results2.fold_balanced_acc.append(ba)
        results2.fold_cms.append(cm)
        results2.models.append(m)

        if verbose:
            print(f"  Fold {fold_idx+1} | F1={f1:.4f} | Bal.Acc={ba:.4f}")

    results2.oof_true   = oof_t2
    results2.oof_pred   = oof_p2
    results2.oof_proba  = oof_pb2
    results2.oof_regime = reg_ns

    if verbose:
        print(f"\nStage 1 OOF | Macro F1: {results1.mean_macro_f1:.4f} ± {results1.std_macro_f1:.4f}")
        print(f"Stage 2 OOF | Macro F1: {results2.mean_macro_f1:.4f} ± {results2.std_macro_f1:.4f}")

    return results1, results2