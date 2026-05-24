"""
ml_predictor/run_baseline.py
==================
3 classifiers × 21 feature groups — window-comparative instability benchmarking.

Usage
-----
  # From project root:
  python -m ml_predictor.run_baseline --data data/dataset3/metadata3.csv

  # Quick smoke test on synthetic data (no real dataset needed):
  python -m ml_predictor.run_baseline --smoke-test

  # Run a single model and feature group:
  python -m ml_predictor.run_baseline --smoke-test --model lightgbm --feature-group B20+

  # Full pipeline with all post-baseline analyses (Items 2–6):
  python -m ml_predictor.run_baseline --data data/dataset3/metadata3.csv \\
      --tune-catboost --ood-holdout --megno-audit --regime-confusion --calibrate

  # Smoke test with all analyses (no real dataset needed):
  python -m ml_predictor.run_baseline --smoke-test \\
      --ood-holdout --regime-confusion --calibrate

Pipeline
--------
  1. Load metadata CSV (or generate synthetic data for smoke test)
  2. Group D feature count verification (Item 1)
  3. Run leakage sanity check (dt, n_steps → outcome predictability)
  3b. Regime sample count guard — flag compact_equal low-n before any CV (Fix 2)
  4. For each model × feature group:
       a. Train with stratified CV (train_model_cv)
       b. Evaluate OOF predictions (evaluate)
       c. Optional two-stage comparison (LightGBM only)
       d. Save per-group plots
  5. Flag unstable CatBoost runs (Item 2) — mark std > 0.015 in CSV
     Also flag CatBoost C-group runs as editorially excluded (Fix 7, in evaluator)
  6. Save all_results.csv  — one row per (model, feature_group)
  7. Post-baseline analyses (run only when their flag is set):
       --tune-catboost   : CatBoost stability grid search (Item 2)
       --ood-holdout     : OOD regime-holdout test for v3_frac (Item 3) — Fix 4
       --megno-audit     : MEGNO borderline audit on B30 OOF (Item 4) — Fix 3
       --regime-confusion: Regime-stratified confusion matrices B5+B30 (Item 5)
       --calibrate       : ECE + isotonic calibration on B30 OOF (Item 6)

OOF caching
-----------
  Items 4, 5, and 6 all need LightGBM B30 OOF predictions. The baseline
  loop produces these as part of the normal run. They are cached in memory
  and reused — B30 is trained exactly once regardless of which post-baseline
  flags are set. Similarly B5 OOF is cached for Item 5.

  If --model or --feature-group restricts the baseline loop such that B5 or
  B30 LightGBM runs are not included, the affected post-baseline analysis
  runs a targeted single CV to generate the required OOF predictions.

Feature groups
--------------
  Baselines  : A, A+
  Core       : B5, B10, B15, B20, B25, B30
               B5+, B10+, B15+, B20+, B25+, B30+
  Window-only: C5, C10, C15, C20, C25, C30
  Upper bound: D  (57 features — IC + IC_eng + w5–w30 + w30_eng)

  The B-series and C-series allow direct comparison of how much predictive
  signal accumulates as the observation window grows from 5% to 30% of the
  inner orbital period. The + suffix adds IC-engineered features and a
  close_encounter_strength term for the window.

Interpreting results
--------------------
  Macro F1 > 0.70   : Strong early-time signal — model learning real physics
  Macro F1 0.50–0.70: Moderate signal — some regimes near chaos horizon
  Macro F1 < 0.50   : Poor signal — check leakage, window definition,
                       or class imbalance within regimes

  Compare models within the same feature group to isolate model capacity.
  Compare feature groups within the same model to isolate information content.
  Comparing B5 → B30 within one model reveals how quickly the instability
  signal saturates — rapid plateau suggests chaos onset at first close encounter.
  The gap between BN and BN+ quantifies the marginal value of physics-engineered
  features at each window size.

Notes on synthetic data (smoke test)
-------------------------------------
  The smoke test generates synthetic data approximating the real three-body
  dataset structure: four regimes with distinct class distributions, physical-
  scale features with realistic ranges, and all six window sizes (w5–w30).
  NOT physically accurate — only exists to verify the pipeline runs
  end-to-end without a real simulation dataset.

  MEGNO is not present in synthetic data. --megno-audit is silently skipped
  when smoke-test mode is active and the MEGNO column is absent.
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running as `python -m ml_predictor.run_baseline` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_predictor.features  import (
    describe_features,
    get_feature_matrix,
    get_labels,
    verify_group_d_count,
    IC_FEATURES,
    W5_FEATURES,
    W30_FEATURES,
)
from ml_predictor.trainer   import (
    train_model_cv,
    train_two_stage_cv,
    tune_catboost_stability,
    MODEL_REGISTRY,
    sample_weights_from_labels,
    N_CLASSES,
    EARLY_STOPPING_ROUNDS,
)
from ml_predictor.evaluator import (
    evaluate,
    leakage_sanity_check,
    save_results_csv,
    save_two_stage_csv,       # Fix 6 — two-stage persistence
    flag_unstable_runs,
    megno_borderline_audit,
    check_regime_sample_counts,  # Fix 2 — compact_equal guard
    compute_ece,
)
from ml_predictor.visualiser import (
    plot_all,
    plot_regime_confusion_grid,
    plot_calibration_before_after,
    plot_megno_audit,
)


# ── Synthetic data generator (smoke test only) ───────────────────────────────

def _make_synthetic_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic three-body-like dataset for pipeline testing.

    Reflects the full feature set across all six windows (w5–w30):
      IC:  q12, q13, q23, r12_init, r3_sep, v3_frac, v3_angle,
           epsilon_total, h_total, M_total, e_inner
      DYN: dE_max_{wN}, e12/13/23_std_{wN}, r_min_12/13/23_{wN}
           for wN in {w5, w10, w15, w20, w25, w30}
           (dL_max excluded — unreliable under discrete sampling)

    Class distributions per regime match expected physics:
      hierarchical  : 70% stable,   10% unstable,  20% chaotic
      asymmetric    : 40% stable,   25% unstable,  35% chaotic
      compact_equal : 15% stable,   45% unstable,  40% chaotic
      scatter       : 25% stable,   40% unstable,  35% chaotic

    Signal grows monotonically with window size — longer windows expose
    more close encounters, concentrating the chaotic signature.
    """
    rng = np.random.default_rng(seed)

    regime_specs = {
        "hierarchical":  {"n": n // 4,        "p": [0.70, 0.10, 0.20]},
        "asymmetric":    {"n": n // 4,        "p": [0.40, 0.25, 0.35]},
        "compact_equal": {"n": n // 4,        "p": [0.15, 0.45, 0.40]},
        "scatter":       {"n": n - 3*(n//4),  "p": [0.25, 0.40, 0.35]},
    }

    # Signal scale factors per window — longer window → stronger separation
    # between stable and chaotic. Mimics accumulation of close encounters.
    _window_signal: dict[str, float] = {
        "w5":  1.0,
        "w10": 1.2,
        "w15": 1.4,
        "w20": 1.6,
        "w25": 1.75,
        "w30": 1.9,
    }

    rows = []
    for regime, spec in regime_specs.items():
        n_reg  = spec["n"]
        labels = rng.choice([0, 1, 2], size=n_reg, p=spec["p"])

        for label in labels:
            r12 = rng.uniform(0.5, 4.0)
            r3  = r12 * rng.uniform(1.0 if label == 1 else 3.0, 8.0)

            r_min_scale = {0: 0.8,  1: 0.15, 2: 0.4 }[label]
            e_std_scale = {0: 0.02, 1: 0.08, 2: 0.15}[label]
            dE_scale    = {0: 1e-5, 1: 5e-4, 2: 1e-4}[label]

            row: dict = {
                # Meta
                "idx":    len(rows),
                "regime": regime,

                # IC features
                "epsilon_total": rng.uniform(-2.0, -0.1),
                "h_total":       rng.uniform(0.1, 5.0),
                "q12":           rng.uniform(0.1, 1.0),
                "q13":           rng.uniform(0.01, 1.0),
                "q23":           rng.uniform(0.01, 1.0),
                "M_total":       rng.uniform(1.0, 3.0),
                "m1":            rng.uniform(0.1, 1.0),
                "m2":            rng.uniform(0.1, 1.0),
                "m3":            rng.uniform(0.1, 1.0),
                "r12_init":      r12,
                "r3_sep":        r3,
                "e_inner":       rng.uniform(0.0, 0.8),
                "v3_frac":       rng.uniform(0.4, 1.35),
                "v3_angle":      rng.uniform(0.0, 2 * np.pi),

                # Labels
                "outcome_class":  label,
                "outcome":        ["stable", "unstable", "chaotic"][label],
                "outcome_class4": label,
            }

            # Window features — signal grows with window size
            for window, scale in _window_signal.items():
                row[f"dE_max_{window}"]    = abs(rng.normal(0, dE_scale * scale))
                row[f"e12_std_{window}"]   = abs(rng.normal(0, e_std_scale * scale)) + 0.005
                row[f"e13_std_{window}"]   = abs(rng.normal(0, e_std_scale * scale)) + 0.005
                row[f"e23_std_{window}"]   = abs(rng.normal(0, e_std_scale * scale)) + 0.005
                row[f"r_min_12_{window}"]  = max(r12 * rng.uniform(0.3, 1.0) * r_min_scale, 0.01)
                row[f"r_min_13_{window}"]  = max(r3  * rng.uniform(0.1, 0.5) * r_min_scale, 0.01)
                row[f"r_min_23_{window}"]  = max(r3  * rng.uniform(0.1, 0.5) * r_min_scale, 0.01)

            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ── OOF cache helpers ─────────────────────────────────────────────────────────

def _get_lgbm_oof(
    df:            pd.DataFrame,
    feature_group: str,
    n_folds:       int,
    cache:         dict,
    verbose:       bool = True,
) -> object:  # returns CVResults
    """
    Return LightGBM CVResults for the given feature group, training only
    if not already present in the cache dict.

    The cache key is ("lightgbm", feature_group). This ensures B30 and B5
    are each trained exactly once even when multiple post-baseline analyses
    need them.

    Parameters
    ----------
    cache : Mutable dict passed from run_baseline — maps
            (model_name, feature_group) → CVResults.
    """
    key = ("lightgbm", feature_group)
    if key not in cache:
        print(f"\n  [OOF cache] Training lightgbm | group {feature_group} "
              f"(needed for post-baseline analysis) ...")
        results = train_model_cv(
            df,
            model_name    = "lightgbm",
            n_folds       = n_folds,
            feature_group = feature_group,
            verbose       = verbose,
        )
        cache[key] = results
    return cache[key]


# ── Post-baseline analysis functions ─────────────────────────────────────────
# Each function is self-contained: it takes df, output_dir, the OOF cache,
# and any analysis-specific params. It saves its own outputs and returns
# nothing — side effects are the contract.

def _run_catboost_tuning(
    df:         pd.DataFrame,
    output_dir: Path,
    n_folds:    int,
    verbose:    bool,
) -> None:
    """
    Item 2 — CatBoost stability grid search on B20, B25, B30.

    Runs tune_catboost_stability() on B20 to find the best params, then
    verifies those same params achieve std <= 0.015 on B25 and B30.
    Saves catboost_params.json (attestation artefact 2b).
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from catboost import CatBoostClassifier, Pool

    tuned_dir = output_dir / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Item 2: CatBoost Stability Tuning ──")

    # Step 1: Grid search on B20 (most problematic group at 17 features)
    best_params = tune_catboost_stability(
        df            = df,
        feature_group = "B20",
        n_folds       = n_folds,
        std_target    = 0.012,
        output_path   = tuned_dir / "catboost_params.json",
        verbose       = verbose,
    )

    # Step 2: Verify the best params hold on B25 and B30
    print("\n  Verifying best params on B25 and B30 ...")
    from sklearn.model_selection import StratifiedKFold

    for verify_group in ["B25", "B30"]:
        X   = get_feature_matrix(df, feature_group=verify_group)
        y   = get_labels(df).values
        X_arr = X.values.astype(np.float32)

        min_count   = np.bincount(y)[np.bincount(y) > 0].min()
        safe_splits = max(2, min(n_folds, min_count))
        skf         = StratifiedKFold(n_splits=safe_splits,
                                      shuffle=True, random_state=42)
        scores = []

        for tr, va in skf.split(X_arr, y):
            sw    = sample_weights_from_labels(y[tr])
            model = CatBoostClassifier(**best_params)
            model.fit(
                Pool(X_arr[tr], y[tr], weight=sw),
                eval_set = Pool(X_arr[va], y[va]),
                verbose  = False,
            )
            preds = model.predict(X_arr[va]).astype(int).flatten()
            scores.append(f1_score(y[va], preds,
                                   average="macro", zero_division=0))

        mean_f1 = float(np.mean(scores))
        std_f1  = float(np.std(scores))
        status  = "✓ stable" if std_f1 <= 0.015 else "⚠ UNSTABLE"
        print(f"  {verify_group}: mean_F1={mean_f1:.4f}  std={std_f1:.4f}  {status}")


def _run_ood_holdout(
    df:         pd.DataFrame,
    output_dir: Path,
    verbose:    bool,
) -> None:
    """
    Item 3 — OOD regime-holdout test for v3_frac.  [Fix 4]

    Trains LightGBM on 3 regimes, tests on the 4th, for groups A (IC only),
    B5 (IC + w5), and B30 (IC + w30). Records macro F1, balanced accuracy,
    v3_frac importance rank, and dummy baselines in each holdout.
    Saves ood_holdout_results.csv and prints the claim sentence.

    Dummy baselines
    ---------------
    Two dummy classifiers are computed per (group, holdout) pair:
      dummy_random_f1    : DummyClassifier(strategy="uniform") — random chance
      dummy_majority_f1  : DummyClassifier(strategy="most_frequent") — trivial baseline

    These provide the lower-bound reference for interpreting LightGBM OOD F1.
    Without them, "macro_f1=0.45 in a holdout" is ambiguous — it may simply
    reflect the class-imbalance floor. A model that doesn't exceed the majority
    baseline is learning the marginal distribution, not the physical instability
    criterion.

    The v3_frac claim sentence now requires the model to exceed the majority
    dummy AND achieve rank ≤ 3. The orbit-velocity ratio v3_frac is a direct
    proxy for whether the outer body exceeds the escape velocity of the inner
    binary (v3_frac ≥ √2 implies near-certain ejection in isolated two-body
    kinematics). Its importance rank in OOD tests directly measures whether
    this physical boundary generalises across regime boundaries.

    Physical motivation
    -------------------
    The scatter regime is defined as "outer body near escape velocity",
    which means high v3_frac by design. Standard 5-fold CV always trains
    on scatter examples so the model trivially learns v3_frac ≈ √2 →
    ejection. In a held-out-regime test this shortcut is unavailable.
    If v3_frac rank drops to > 3 in any holdout, the strong-claim sentence
    must be replaced by the qualified-claim sentence.

    B30 inclusion
    -------------
    Adding B30 reveals whether the 30%-window close-encounter signal (dE_max,
    r_min, e_std) provides OOD robustness beyond what IC geometry alone
    achieves in group A. If B30 OOD F1 ≥ B5 OOD F1, the encounter dynamics
    encode generalisable physics. If B30 degrades, the window features are
    fitting regime-specific encounter patterns that don't transfer.
    """
    import lightgbm as lgb
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score, balanced_accuracy_score

    print("\n── Item 3: OOD Regime-Holdout Test (v3_frac) ──")

    if "regime" not in df.columns:
        print("  [skip] 'regime' column not found in dataset.")
        return

    REGIMES = ["hierarchical", "asymmetric", "compact_equal", "scatter"]
    missing_regimes = [r for r in REGIMES if r not in df["regime"].values]
    if missing_regimes:
        print(f"  [skip] Regimes not found in data: {missing_regimes}")
        return

    # Fix 4: B30 added alongside A and B5
    GROUPS = {
        "A":   IC_FEATURES,
        "B5":  IC_FEATURES + W5_FEATURES,
        "B30": IC_FEATURES + W30_FEATURES,
    }

    lgbm_params = {
        "n_estimators":  500,
        "learning_rate": 0.05,
        "num_leaves":    63,
        "random_state":  42,
        "verbose":      -1,
        "n_jobs":       -1,
    }

    all_results = []

    for group_name, feats in GROUPS.items():
        # Only use features that actually exist in this DataFrame
        available_feats = [f for f in feats if f in df.columns]
        if not available_feats:
            print(f"  [skip] No features available for group {group_name}")
            continue

        X = df[available_feats]
        y = get_labels(df)

        for holdout in REGIMES:
            train_mask = df["regime"] != holdout
            test_mask  = df["regime"] == holdout

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask],  y[test_mask]

            if len(X_te) == 0:
                continue

            # ── LightGBM model ────────────────────────────────────────────────
            model = lgb.LGBMClassifier(**lgbm_params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            mf1  = f1_score(y_te, preds, average="macro", zero_division=0)
            bacc = balanced_accuracy_score(y_te, preds)

            # v3_frac importance rank (split-count based)
            imp_dict     = dict(zip(available_feats, model.feature_importances_))
            sorted_feats = sorted(imp_dict, key=imp_dict.get, reverse=True)
            v3_rank      = (sorted_feats.index("v3_frac") + 1
                            if "v3_frac" in sorted_feats else -1)
            v3_imp       = imp_dict.get("v3_frac", 0.0)

            # ── Fix 4: Dummy baselines ────────────────────────────────────────
            # Both dummies are fitted on the same training split to avoid
            # any leakage of test-set class distribution.
            dummy_random = DummyClassifier(strategy="uniform", random_state=42)
            dummy_random.fit(X_tr, y_tr)
            dummy_random_f1 = float(f1_score(
                y_te, dummy_random.predict(X_te), average="macro", zero_division=0
            ))

            dummy_majority = DummyClassifier(strategy="most_frequent")
            dummy_majority.fit(X_tr, y_tr)
            dummy_majority_f1 = float(f1_score(
                y_te, dummy_majority.predict(X_te), average="macro", zero_division=0
            ))

            row = {
                "group":               group_name,
                "holdout_regime":      holdout,
                "train_n":             int(train_mask.sum()),
                "test_n":              int(test_mask.sum()),
                "macro_f1":            round(float(mf1),  4),
                "bal_acc":             round(float(bacc), 4),
                "v3_frac_importance":  round(float(v3_imp), 1),
                "v3_frac_rank":        v3_rank,
                # Fix 4: dummy columns
                "dummy_random_f1":     round(dummy_random_f1,  4),
                "dummy_majority_f1":   round(dummy_majority_f1, 4),
            }
            all_results.append(row)

            if verbose:
                print(f"  {group_name} | holdout={holdout:<18} | "
                      f"macro_F1={mf1:.4f} | "
                      f"dummy_maj={dummy_majority_f1:.4f} | "
                      f"v3_frac rank={v3_rank}")

    if not all_results:
        print("  [skip] No holdout results generated.")
        return

    results_df = pd.DataFrame(all_results)
    out_path   = output_dir / "ood_holdout_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # ── Claim sentence ────────────────────────────────────────────────────────
    # Fix 4: claim requires (a) rank ≤ 3 AND (b) model exceeds majority dummy.
    # A model that ranks v3_frac top-3 but doesn't beat the majority baseline
    # is not making a meaningful physical claim — it's recovering the class
    # distribution of the training set, not the instability criterion.
    max_rank = results_df["v3_frac_rank"].replace(-1, 999).max()
    beats_dummy = (results_df["macro_f1"] > results_df["dummy_majority_f1"]).all()

    if max_rank <= 3 and beats_dummy:
        claim = (
            "v3_frac is a physically robust discriminator of three-body fate "
            "across dynamical regimes (rank ≤ 3 in all regime-holdout experiments, "
            "and LightGBM exceeds the majority-class dummy in all holdouts)."
        )
    elif max_rank > 3:
        worst_row = results_df.loc[
            results_df["v3_frac_rank"].replace(-1, 999).idxmax()
        ]
        claim = (
            f"v3_frac is strongly discriminative within the sampled phase space "
            f"but its rank drops to {max_rank} when holding out the "
            f"'{worst_row['holdout_regime']}' regime (group {worst_row['group']}). "
            f"Generalisation to regimes with different orbital energy distributions "
            f"requires further validation."
        )
    else:
        # rank ≤ 3 but model doesn't always beat dummy — weak signal
        n_fail = (results_df["macro_f1"] <= results_df["dummy_majority_f1"]).sum()
        claim = (
            f"v3_frac achieves rank ≤ 3 in all holdouts but LightGBM fails to "
            f"exceed the majority-class dummy in {n_fail} holdout(s). "
            f"The v3_frac signal is present but insufficient to lift overall OOD "
            f"performance above the class-distribution baseline."
        )

    print(f"\n  Claim sentence:\n  → {claim}")

    # Print the filled table for the paper
    print("\n  OOD holdout table:")
    print(results_df.to_string(index=False))


def _run_megno_audit(
    df:         pd.DataFrame,
    output_dir: Path,
    oof_cache:  dict,
    n_folds:    int,
    verbose:    bool,
) -> None:
    """
    Item 4 — MEGNO borderline audit on LightGBM B30 OOF predictions.  [Fix 3]

    Requires MEGNO column in df. Silently skipped if absent (smoke-test mode).

    Fix 3: Runs the audit at three thresholds — 3.0, 4.0, and 5.0 — to test
    sensitivity of the labelling-ambiguity interpretation. The canonical
    labelling threshold is 3.0 (matching the dataset). Thresholds 4.0 and 5.0
    serve as sensitivity checks: if the chaotic→stable confusion concentration
    near [threshold, threshold+1.0] persists at higher thresholds, the
    interpretation is robust — the confusion tracks the instability boundary
    regardless of where we draw it, consistent with slow Lyapunov divergence
    rather than a systematic model failure.

    Saves
    -----
    oof_B30_with_megno.csv               — OOF predictions joined with MEGNO
    megno_audit_B30_t3.0.png             — MEGNO histogram at threshold 3.0
    megno_audit_B30_t4.0.png             — MEGNO histogram at threshold 4.0
    megno_audit_B30_t5.0.png             — MEGNO histogram at threshold 5.0
    megno_threshold_sensitivity.csv      — bin counts for all three thresholds
    """
    print("\n── Item 4: MEGNO Borderline Audit ──")

    if "MEGNO" not in df.columns:
        print("  [skip] MEGNO column not found — "
              "unavailable in smoke-test mode or dataset missing column.")
        return

    # Get B30 OOF predictions (cached from baseline loop if available)
    b30_results = _get_lgbm_oof(df, "B30", n_folds, oof_cache, verbose)

    # Build audit DataFrame: join OOF predictions with MEGNO from df.
    # df index and OOF arrays are aligned — both follow the original row order.
    oof_df = pd.DataFrame({
        "true_class": b30_results.oof_true,
        "pred_class": b30_results.oof_pred,
        "MEGNO":      df["MEGNO"].values,
        "regime":     df["regime"].values if "regime" in df.columns
                      else ["unknown"] * len(df),
    })

    # Save oof_B30_with_megno.csv (attestation 4a)
    megno_oof_path = output_dir / "oof_B30_with_megno.csv"
    oof_df.to_csv(megno_oof_path, index=False)
    print(f"  Saved: {megno_oof_path}")

    # Fix 3: Run audit at all three thresholds; accumulate bin-count rows
    # for the combined sensitivity CSV.
    sensitivity_rows = []

    for threshold in [3.0, 4.0, 5.0]:
        print(f"\n  -- Threshold {threshold:.1f} --")
        audit_df = megno_borderline_audit(
            oof_df,
            megno_threshold = threshold,
            verbose         = verbose,
        )

        # Tag each row with the threshold so they can be concatenated
        audit_df["megno_threshold"] = threshold
        sensitivity_rows.append(audit_df)

        # Save per-threshold audit plot (attestation 4b — one file per threshold)
        plot_megno_audit(
            oof_df,
            group_name      = f"B30",
            output_dir      = output_dir,
            megno_threshold = threshold,
        )

    # Save combined sensitivity table (attestation 4c)
    sensitivity_df = pd.concat(sensitivity_rows, ignore_index=True)
    sens_path = output_dir / "megno_threshold_sensitivity.csv"
    sensitivity_df.to_csv(sens_path, index=False)
    print(f"\n  Saved: {sens_path}")


def _run_regime_confusion(
    df:         pd.DataFrame,
    output_dir: Path,
    oof_cache:  dict,
    n_folds:    int,
    verbose:    bool,
) -> None:
    """
    Item 5 — Regime-stratified confusion matrices for B5 and B30.

    Saves oof_preds_B5.csv, oof_preds_B30.csv (attestation 5a) and
    lgb_B5_confusion_by_regime.png, lgb_B30_confusion_by_regime.png (5b).
    Prints per-regime chaotic→stable rates (5c) and the written answer (5d).
    """
    print("\n── Item 5: Regime-Stratified Confusion Matrices ──")

    if "regime" not in df.columns:
        print("  [skip] 'regime' column not found in dataset.")
        return

    for group_name in ["B5", "B30"]:
        results = _get_lgbm_oof(df, group_name, n_folds, oof_cache, verbose)

        # Build flat OOF DataFrame with regime labels
        oof_df = pd.DataFrame({
            "true":   results.oof_true,
            "pred":   results.oof_pred,
            "regime": df["regime"].values,
        })

        # Save OOF preds CSV (attestation 5a)
        preds_path = output_dir / f"oof_preds_{group_name}.csv"
        oof_df.to_csv(preds_path, index=False)
        print(f"  Saved: {preds_path}")

        # Plot 2×2 regime confusion grid (attestation 5b, 5c)
        plot_regime_confusion_grid(
            oof_df     = oof_df,
            group_name = group_name,
            output_dir = output_dir,
        )

    # ── Written answer (attestation 5d) ──────────────────────────────────────
    # Load B30 results (already cached) and compute per-regime chaotic→stable
    # rate to identify the dominant driver.
    b30_results = oof_cache.get(("lightgbm", "B30"))
    if b30_results is not None and "regime" in df.columns:
        from sklearn.metrics import confusion_matrix as sk_cm

        oof_b30 = pd.DataFrame({
            "true":   b30_results.oof_true,
            "pred":   b30_results.oof_pred,
            "regime": df["regime"].values,
        })

        regime_cs = {}
        for regime in df["regime"].unique():
            sub = oof_b30[oof_b30["regime"] == regime]
            if len(sub) == 0:
                continue
            cm  = sk_cm(sub["true"], sub["pred"],
                        labels=[0, 1, 2], normalize="true")
            regime_cs[regime] = float(cm[2, 0])   # chaotic→stable rate

        if regime_cs:
            worst_regime = max(regime_cs, key=regime_cs.get)
            worst_rate   = regime_cs[worst_regime]
            total_cs     = sum(regime_cs.values())
            pct_driven   = (worst_rate / total_cs * 100) if total_cs > 0 else 0.0

            written_answer = (
                f"The '{worst_regime}' regime drives the aggregate chaotic→stable "
                f"confusion (normalised rate {worst_rate:.3f}, "
                f"{pct_driven:.0f}% of total confusion across regimes on B30)."
            )
            print(f"\n  Written answer (attestation 5d):\n  → {written_answer}")


def _run_calibration(
    df:         pd.DataFrame,
    output_dir: Path,
    oof_cache:  dict,
    n_folds:    int,
    verbose:    bool,
) -> None:
    """
    Item 6 — ECE computation and isotonic calibration on LightGBM B30.

    Saves oof_probs_B30.csv (attestation 6a),
    lgb_B30_calibration_before_after.png (6b),
    ece_results_B30.csv (6c).
    Prints ECE interpretation sentences per class (6d).
    """
    print("\n── Item 6: Calibration + ECE (LightGBM B30) ──")

    b30_results = _get_lgbm_oof(df, "B30", n_folds, oof_cache, verbose)

    # Build probability DataFrame (attestation 6a)
    prob_df = pd.DataFrame(
        b30_results.oof_proba,
        columns=["prob_stable", "prob_unstable", "prob_chaotic"],
    )
    prob_df["true"]   = b30_results.oof_true
    prob_df["pred"]   = b30_results.oof_pred
    prob_df["regime"] = (df["regime"].values if "regime" in df.columns
                         else ["unknown"] * len(df))

    probs_path = output_dir / "oof_probs_B30.csv"
    prob_df.to_csv(probs_path, index=False)
    print(f"  Saved: {probs_path}")

    # Plot calibration before/after + save ECE CSV (attestations 6b, 6c, 6d)
    plot_calibration_before_after(
        prob_df    = prob_df,
        group_name = "B30",
        output_dir = output_dir,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_baseline(
    data_path:      Path | None = None,
    output_dir:     Path        = Path("results/baseline"),
    n_folds:        int         = 5,
    two_stage:      bool        = True,
    smoke_test:     bool        = False,
    save_plots:     bool        = True,
    verbose:        bool        = True,
    feature_group:  str | None  = None,
    model:          str | None  = None,
    # ── Post-baseline flags (Items 2–6) ───────────────────────────────────────
    tune_catboost:    bool = False,
    ood_holdout:      bool = False,
    megno_audit:      bool = False,
    regime_confusion: bool = False,
    calibrate:        bool = False,
) -> dict:
    """
    Full multi-model baseline pipeline.

    Runs every combination of (model, feature_group) unless restricted by
    the model or feature_group arguments.

    Parameters
    ----------
    data_path        : Path to metadata CSV. Required unless smoke_test=True.
    output_dir       : Root directory for all output (plots, CSVs).
    n_folds          : Stratified CV folds (default 5).
    two_stage        : Run LightGBM two-stage comparison per feature group.
                       Results are saved to two_stage_results.csv (Fix 6).
    smoke_test       : Use synthetic data instead of real dataset.
    save_plots       : Save confusion matrix, calibration, importance plots.
    verbose          : Print per-fold progress and per-run evaluation summary.
    feature_group    : One of the 21 registered groups. Default = run all.
    model            : One of the MODEL_REGISTRY keys. Default = run all.
    tune_catboost    : Run CatBoost stability grid search (Item 2).
    ood_holdout      : Run OOD regime-holdout v3_frac test, groups A/B5/B30
                       with dummy baselines (Item 3, Fix 4).
    megno_audit      : Run MEGNO borderline audit on B30 OOF at thresholds
                       3.0, 4.0, 5.0 (Item 4, Fix 3).
    regime_confusion : Save regime-stratified confusion matrices B5+B30 (Item 5).
    calibrate        : Run ECE + isotonic calibration on B30 OOF (Item 6).

    Returns
    -------
    dict keyed by (model_name, feature_group) → {results, report, summary}
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

    # ── 2. Group D verification (Item 1) ──────────────────────────────────────
    print("\n── Item 1: Group D Feature Count Verification ──")
    try:
        verify_group_d_count(df, verbose=True)
    except AssertionError as e:
        print(f"  ⚠  {e}")
        print("  Continuing — some window columns may be absent in this dataset.")

    # ── 3. Leakage check ──────────────────────────────────────────────────────
    print("\n── Leakage check ──")
    leakage_result = leakage_sanity_check(df, verbose=True)

    # ── 3b. Fix 2: Regime sample count guard ──────────────────────────────────
    # compact_equal has ~11 samples in the real dataset — 5-fold CV will
    # produce degenerate splits (≤ 2 samples/fold per class). Flag before
    # training so regime-confusion panels can be annotated appropriately.
    print("\n── Regime sample count guard ──")
    regime_count_result = check_regime_sample_counts(
        df,
        min_samples_per_fold = 5,
        n_folds              = n_folds,
        verbose              = True,
    )
    if regime_count_result["any_flagged"]:
        flagged = regime_count_result["flagged_regimes"]
        print(
            f"\n  ⚠  Regimes {flagged} have < 5 samples/fold in at least one class.\n"
            f"  CV statistics for these regimes are unreliable. Confusion matrix\n"
            f"  panels will be labelled '⚠ low-n' in regime_confusion plots.\n"
            f"  Do NOT make quantitative claims about per-class performance within\n"
            f"  these regimes — a single misclassification moves recall by ≥ 50%."
        )

    # describe_features is a data integrity checkpoint — run silently,
    # result available via return value but not printed to avoid clutter.
    describe_features(df, verbose=False)

    # ── 4. Resolve run scope ──────────────────────────────────────────────────
    ALL_GROUPS = [
        # Baselines
        "A", "A+",
        # Core — no engineering
        "B5", "B10", "B15", "B20", "B25", "B30",
        # Core — with engineering
        "B5+", "B10+", "B15+", "B20+", "B25+", "B30+",
        # Window-only
        "C5", "C10", "C15", "C20", "C25", "C30",
        # Upper bound
        "D",
    ]
    ALL_MODELS = list(MODEL_REGISTRY.keys())

    groups_to_run = [feature_group] if feature_group is not None else ALL_GROUPS
    models_to_run = [model]         if model         is not None else ALL_MODELS

    n_runs = len(groups_to_run) * len(models_to_run)
    print(f"\n── Baseline: {len(models_to_run)} models × "
          f"{len(groups_to_run)} feature groups = {n_runs} runs ──")

    all_outputs    = {}
    all_reports    = []
    two_stage_rows = []   # Fix 6: accumulated per-group two-stage stats

    # OOF cache: (model_name, feature_group) → CVResults
    # Populated by the baseline loop and consumed by post-baseline analyses.
    # Avoids re-training B5/B30 when multiple post-baseline flags are set.
    oof_cache: dict = {}

    # ── 5. Baseline loop ──────────────────────────────────────────────────────
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

            # Cache all LightGBM results for downstream reuse
            if model_name == "lightgbm":
                oof_cache[("lightgbm", fg)] = results

            report = evaluate(
                results,
                df            = df,
                model_name    = model_name,
                feature_group = fg,
                verbose       = verbose,
                # Fix 5: evaluate() already calls check_f1_discrepancy when
                # verbose=True — no additional call needed here.
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

                direct_ba = float(np.mean(results.fold_balanced_acc))
                s1_ba     = float(np.mean(s1_results.fold_balanced_acc))
                s2_ba     = float(np.mean(s2_results.fold_balanced_acc))
                direct_f1 = report.macro_f1
                s1_f1     = s1_results.mean_macro_f1
                s2_f1     = s2_results.mean_macro_f1

                if verbose:
                    print(f"  Two-stage bal.acc — direct: {direct_ba:.4f} | "
                          f"stage1: {s1_ba:.4f} | stage2: {s2_ba:.4f}")

                # Fix 6: accumulate row for two_stage_results.csv
                two_stage_rows.append({
                    "feature_group": fg,
                    "direct_ba":     round(direct_ba, 4),
                    "s1_ba":         round(s1_ba,     4),
                    "s2_ba":         round(s2_ba,     4),
                    "direct_f1":     round(direct_f1, 4),
                    "s1_f1":         round(s1_f1,     4),
                    "s2_f1":         round(s2_f1,     4),
                })

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

    # ── 6. Save results CSV + stability guard (Items 1d, 2) ──────────────────
    print("\n── Saving baseline results ──")
    df_all = save_results_csv(all_reports, output_dir=output_dir)

    # Flag any CatBoost runs with std > 0.015 (Item 2) and CatBoost C-groups
    # (Fix 7 — window-only features provide no meaningful splits for symmetric
    # trees; these rows are editorially excluded regardless of std).
    df_all = flag_unstable_runs(df_all, std_threshold=0.015, verbose=True)
    # Overwrite all_results.csv with stability flags included
    df_all.to_csv(output_dir / "all_results.csv", index=False)

    # Fix 5: Show fold_macro_f1_mean alongside macro_f1 in the leaderboard so
    # any pooled-OOF vs mean-per-fold discrepancy is immediately visible.
    print("\n── Leaderboard (macro F1, stable runs only) ──")
    display_cols = [
        "model", "feature_group",
        "macro_f1",          # pooled OOF — canonical
        "fold_macro_f1_mean",# mean of per-fold F1s — discrepancy cross-check
        "macro_f1_std",
        "balanced_acc", "mean_brier", "stable_run",
    ]
    # Only show columns that exist (fold_macro_f1_mean present after save_results_csv)
    display_cols = [c for c in display_cols if c in df_all.columns]
    print(
        df_all[display_cols]
        .to_string(index=False)
    )

    # Fix 6: Persist two-stage comparison results whenever two_stage=True.
    # save_two_stage_csv handles the empty-list case gracefully.
    if two_stage and two_stage_rows:
        print("\n── Two-stage vs Direct (LightGBM) ──")
        save_two_stage_csv(two_stage_rows, output_dir=output_dir)

    # ── 7. Post-baseline analyses ─────────────────────────────────────────────
    any_post = tune_catboost or ood_holdout or megno_audit or regime_confusion or calibrate
    if any_post:
        print("\n── Post-Baseline Analyses ──")

    if tune_catboost:
        _run_catboost_tuning(df, output_dir, n_folds, verbose)

    if ood_holdout:
        _run_ood_holdout(df, output_dir, verbose)

    if megno_audit:
        _run_megno_audit(df, output_dir, oof_cache, n_folds, verbose)

    if regime_confusion:
        _run_regime_confusion(df, output_dir, oof_cache, n_folds, verbose)

    if calibrate:
        _run_calibration(df, output_dir, oof_cache, n_folds, verbose)

    return all_outputs


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Multi-model three-body instability baseline — window comparative study.\n"
            "Post-baseline analyses (Items 2–6) are opt-in via flags below."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Existing args (unchanged) ─────────────────────────────────────────────
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
                   help="Feature group to run. One of: A, A+, "
                        "B5–B30, B5+–B30+, C5–C30, D. "
                        "Default = run all 21 groups.")
    p.add_argument("--model",         type=str, default=None,
                   help=f"Model to run. One of: {list(MODEL_REGISTRY.keys())}. "
                        "Default = run all.")

    # ── New flags (Items 2–6) ─────────────────────────────────────────────────
    post = p.add_argument_group(
        "Post-baseline analyses (Action List Items 2–6)",
        "Each flag runs one attestation item after the baseline loop. "
        "All outputs are saved to --output. B5 and B30 LightGBM OOF "
        "predictions are cached from the baseline loop and reused — "
        "no redundant training.",
    )
    post.add_argument(
        "--tune-catboost",
        action="store_true",
        help=(
            "[Item 2] CatBoost stability grid search on B20/B25/B30. "
            "Saves catboost_params.json. Runs after the baseline loop."
        ),
    )
    post.add_argument(
        "--ood-holdout",
        action="store_true",
        help=(
            "[Item 3, Fix 4] OOD regime-holdout test for v3_frac on groups "
            "A, B5, and B30. Includes dummy baselines (random + majority). "
            "Saves ood_holdout_results.csv. Prints claim sentence."
        ),
    )
    post.add_argument(
        "--megno-audit",
        action="store_true",
        help=(
            "[Item 4, Fix 3] MEGNO borderline audit on LightGBM B30 OOF "
            "at thresholds 3.0, 4.0, and 5.0. "
            "Requires MEGNO column in dataset (skipped in smoke-test mode). "
            "Saves oof_B30_with_megno.csv, megno_audit_B30_t{3,4,5}.0.png, "
            "and megno_threshold_sensitivity.csv."
        ),
    )
    post.add_argument(
        "--regime-confusion",
        action="store_true",
        help=(
            "[Item 5] Regime-stratified confusion matrices for B5 and B30. "
            "Saves oof_preds_B5.csv, oof_preds_B30.csv, "
            "lgb_B5_confusion_by_regime.png, lgb_B30_confusion_by_regime.png."
        ),
    )
    post.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "[Item 6] ECE + isotonic calibration on LightGBM B30 OOF. "
            "Saves oof_probs_B30.csv, lgb_B30_calibration_before_after.png, "
            "ece_results_B30.csv."
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_baseline(
        data_path        = args.data,
        output_dir       = args.output,
        n_folds          = args.folds,
        two_stage        = not args.no_two_stage,
        smoke_test       = args.smoke_test,
        save_plots       = not args.no_plots,
        feature_group    = args.feature_group,
        model            = args.model,
        tune_catboost    = args.tune_catboost,
        ood_holdout      = args.ood_holdout,
        megno_audit      = args.megno_audit,
        regime_confusion = args.regime_confusion,
        calibrate        = args.calibrate,
    )