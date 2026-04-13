"""
ml/experiment_a.py
==================
Experiment A — Eccentricity Regression (FIXED)
===============================================

Research question
-----------------
Can physics features (ε, h, μ, q, r₀) predict eccentricity better
than raw features (E₀, L₀, μ, r₀) when tested on unseen mass ratios?

This tests mass-ratio generalisation: train on q ≤ 0.15 (star-planet),
evaluate on q ≥ 0.35 (near-equal-mass binary).

Fixes applied vs. original
---------------------------
1. Quality filter: switched from dE_max (broke parabolic class) to
   dL_max ≤ 1e-6, which works correctly for all orbit types.
2. Removed bound-only filter: regression runs on e ∈ [0, ∞).
   Bound-only lost 500 hyperbolic samples and all context about
   what the model needs to distinguish at the e=1 boundary.
3. A4 now uses only MEGNO_clean, dE_max, dE_slope, dL_max, e_inst_std.
   Removed residual_max/residual_mean (computed from analytical e directly).
4. MEGNO_clean instead of raw MEGNO (clips negatives from convergence artifact).
5. Increased XGBoost depth and estimators for the complex nonlinear function.

Experiments
-----------
  A1  Ridge regression on PHYSICS_NORM         — linear baseline
  A2  XGBoost on PHYSICS_NORM (ε, h, μ, q, r₀) — main model
  A3  XGBoost on PHYSICS_RAW (E₀, L₀, μ, r₀)  — ablation: no normalisation
  A4  XGBoost on ALL_PHYSICS (norm + dynamics)  — with chaos indicators

Usage
-----
    python ml/experiment_a.py
    python ml/experiment_a.py --data data/dataset/metadata.csv
"""

from __future__ import annotations

import sys, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.features import (
    PHYSICS_NORM, PHYSICS_RAW, DYNAMICS, ALL_PHYSICS,
    extract, extract_regression_target,
    train_test_generalisation_split, load_dataset,
    CLASS_NAMES, FEATURE_DESCRIPTIONS,
)
warnings.filterwarnings("ignore")

DEFAULT_CSV = Path("data/dataset/metadata.csv")
OUTPUT_DIR  = Path("outputs/exp_a")
SEED        = 42

# Deeper trees for the nonlinear e = sqrt(1 + 2*eps*h^2/M^2) function
XGB_PARAMS = dict(
    n_estimators     = 800,
    max_depth        = 8,
    learning_rate    = 0.03,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    random_state     = SEED,
    n_jobs           = -1,
    verbosity        = 0,
)

STYLE = dict(bg="#080810", ax="#0d0d1a", grid="#1e1e2e", text="#c4c4d4",
             lab="#7777aa", a="#38bdf8", b="#fb923c", c="#86efac", d="#f87171")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, label):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"    {label:<40}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"label": label, "R2": r2, "RMSE": rmse, "MAE": mae}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(STYLE["ax"])
    ax.set_title(title,  color=STYLE["text"], fontsize=9, pad=4, fontfamily="monospace")
    ax.set_xlabel(xlabel, color=STYLE["lab"], fontsize=8)
    ax.set_ylabel(ylabel, color=STYLE["lab"], fontsize=8)
    ax.tick_params(colors="#666688", labelsize=7.5)
    for sp in ax.spines.values(): sp.set_edgecolor(STYLE["grid"])
    ax.grid(True, color=STYLE["grid"], lw=0.4, alpha=0.7)


def plot_pred_vs_true(y_true, preds: dict, title, path):
    n = len(preds)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), facecolor=STYLE["bg"])
    if n == 1: axes = [axes]
    colors = [STYLE["a"], STYLE["b"], STYLE["c"], STYLE["d"]]
    for ax, (label, y_pred), col in zip(axes, preds.items(), colors):
        r2 = r2_score(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=col)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo,hi],[lo,hi],"w--",lw=0.8,alpha=0.5,label="perfect")
        _style(ax, f"{label}\nR² = {r2:.4f}", "True e", "Predicted e")
        ax.legend(fontsize=7, facecolor="#13131f", labelcolor="#bbbbcc")
    fig.suptitle(title, color=STYLE["text"], fontsize=10, fontfamily="monospace")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_generalisation_curve(results_by_model, title, path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=STYLE["bg"])
    _style(ax, title, "Mass ratio q  (training: q ≤ 0.15)", "RMSE")
    colors = [STYLE["a"], STYLE["b"], STYLE["c"], STYLE["d"]]
    for (name, pts), col in zip(results_by_model.items(), colors):
        qs = [p[0] for p in pts]; vals = [p[1] for p in pts]
        ax.plot(qs, vals, "o-", color=col, lw=1.5, ms=5, label=name)
    ax.axvline(0.15, color="#fde68a", ls="--", lw=0.8, alpha=0.7, label="train boundary")
    ax.legend(fontsize=7, facecolor="#13131f", labelcolor="#bbbbcc")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names, title, path):
    imp = model.feature_importances_
    idx = np.argsort(imp)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, max(3, len(feature_names)*0.4)),
                           facecolor=STYLE["bg"])
    ax.barh([feature_names[i] for i in idx], imp[idx], color=STYLE["a"], alpha=0.85)
    _style(ax, title, "Importance (gain)", "Feature")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residuals_by_e(y_true, y_pred, label, path):
    """Show where the model makes errors — by eccentricity bin."""
    residuals = np.abs(y_true - y_pred)
    bins = np.linspace(0, min(y_true.max(), 5), 12)
    bin_idx = np.digitize(y_true, bins)
    bin_mids, bin_rmse = [], []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() > 5:
            bin_mids.append((bins[b-1] + bins[b]) / 2)
            bin_rmse.append(np.sqrt(np.mean(residuals[mask]**2)))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=STYLE["bg"])
    ax.bar(bin_mids, bin_rmse, width=(bins[1]-bins[0])*0.8,
           color=STYLE["a"], alpha=0.8)
    ax.axvline(1.0, color="#fde68a", ls="--", lw=1.0, label="e=1 boundary")
    _style(ax, f"{label} — RMSE by eccentricity bin", "True e", "RMSE")
    ax.legend(fontsize=7, facecolor="#13131f", labelcolor="#bbbbcc")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(csv_path: Path = DEFAULT_CSV) -> None:
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    print("\n" + "═"*60)
    print("  EXPERIMENT A — Eccentricity Regression")
    print("  Quality filter: dL_max ≤ 1e-6  (all orbit types retained)")
    print("  Target: e ∈ [0, ∞)  (bound AND unbound)")
    print("═"*60)

    # ── Load ──────────────────────────────────────────────────────────────
    df = load_dataset(csv_path,
                      apply_quality_filter=True,
                      dL_max_threshold=1e-6)

    # ── Splits ─────────────────────────────────────────────────────────────
    df_tr, df_id, df_ood = train_test_generalisation_split(df, seed=SEED)
    print(f"\n  Train (q≤0.15):     {len(df_tr):4d}  |  "
          + "  ".join(f"{k}:{v}" for k,v in df_tr["orbit_name"].value_counts().items()))
    print(f"  Test in-dist:       {len(df_id):4d}  |  "
          + "  ".join(f"{k}:{v}" for k,v in df_id["orbit_name"].value_counts().items()))
    print(f"  Test OOD (q≥0.35):  {len(df_ood):4d}  |  "
          + "  ".join(f"{k}:{v}" for k,v in df_ood["orbit_name"].value_counts().items()))

    y_tr  = extract_regression_target(df_tr)
    y_id  = extract_regression_target(df_id)
    y_ood = extract_regression_target(df_ood)

    all_results = []
    trained_models = {}

    # ── A1: Ridge on PHYSICS_NORM ──────────────────────────────────────────
    print("\n── A1: Ridge Regression  (linear baseline) ──")
    pipe_a1 = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    pipe_a1.fit(extract(df_tr, PHYSICS_NORM), y_tr)
    all_results += [
        _metrics(y_id,  pipe_a1.predict(extract(df_id,  PHYSICS_NORM)), "A1 Ridge PHYSICS_NORM | in-dist"),
        _metrics(y_ood, pipe_a1.predict(extract(df_ood, PHYSICS_NORM)), "A1 Ridge PHYSICS_NORM | OOD"),
    ]

    # ── A2: XGBoost on PHYSICS_NORM ───────────────────────────────────────
    print("\n── A2: XGBoost on PHYSICS_NORM (main model) ──")
    m_a2 = xgb.XGBRegressor(**XGB_PARAMS)
    m_a2.fit(extract(df_tr, PHYSICS_NORM), y_tr)
    trained_models["A2 PHYSICS_NORM"] = (PHYSICS_NORM, m_a2)
    all_results += [
        _metrics(y_id,  m_a2.predict(extract(df_id,  PHYSICS_NORM)), "A2 XGB PHYSICS_NORM | in-dist"),
        _metrics(y_ood, m_a2.predict(extract(df_ood, PHYSICS_NORM)), "A2 XGB PHYSICS_NORM | OOD"),
    ]
    joblib.dump(m_a2, out / "models" / "model_a2_physics_norm.joblib")

    # ── A3: XGBoost on PHYSICS_RAW (ablation) ─────────────────────────────
    print("\n── A3: XGBoost on PHYSICS_RAW (ablation — no mass normalisation) ──")
    m_a3 = xgb.XGBRegressor(**XGB_PARAMS)
    m_a3.fit(extract(df_tr, PHYSICS_RAW), y_tr)
    trained_models["A3 PHYSICS_RAW"] = (PHYSICS_RAW, m_a3)
    all_results += [
        _metrics(y_id,  m_a3.predict(extract(df_id,  PHYSICS_RAW)), "A3 XGB PHYSICS_RAW | in-dist"),
        _metrics(y_ood, m_a3.predict(extract(df_ood, PHYSICS_RAW)), "A3 XGB PHYSICS_RAW | OOD"),
    ]

    # ── A4: XGBoost on ALL_PHYSICS ─────────────────────────────────────────
    print("\n── A4: XGBoost on ALL_PHYSICS (physics + chaos indicators) ──")
    print("  Note: MEGNO_clean used (clips negative convergence artifacts).")
    print("  Note: residual_max/mean excluded (computed from analytical e).")
    m_a4 = xgb.XGBRegressor(**XGB_PARAMS)
    m_a4.fit(extract(df_tr, ALL_PHYSICS), y_tr)
    trained_models["A4 ALL_PHYSICS"] = (ALL_PHYSICS, m_a4)
    all_results += [
        _metrics(y_id,  m_a4.predict(extract(df_id,  ALL_PHYSICS)), "A4 XGB ALL_PHYSICS | in-dist"),
        _metrics(y_ood, m_a4.predict(extract(df_ood, ALL_PHYSICS)), "A4 XGB ALL_PHYSICS | OOD"),
    ]
    joblib.dump(m_a4, out / "models" / "model_a4_all_physics.joblib")

    # ── Generalisation curve: RMSE vs q bin ────────────────────────────────
    print("\n── Mass ratio generalisation test ──")
    q_bins = [(0.001,0.05),(0.05,0.15),(0.15,0.25),(0.25,0.35),
              (0.35,0.55),(0.55,0.80),(0.80,1.00)]
    gen_results = {"A2 PHYSICS_NORM": [], "A3 PHYSICS_RAW": [],
                   "A4 ALL_PHYSICS": []}
    for lo, hi in q_bins:
        sub = df[(df["q"] > lo) & (df["q"] <= hi)]
        if len(sub) < 10:
            continue
        q_mid = (lo + hi) / 2
        for name, (feats, mdl) in trained_models.items():
            rmse = np.sqrt(mean_squared_error(
                sub["e"].values, mdl.predict(sub[feats].values)))
            gen_results[name].append((q_mid, rmse))
            print(f"  {name:<22}  q∈({lo:.3f},{hi:.3f}]  "
                  f"n={len(sub):4d}  RMSE={rmse:.4f}")

    # ── Save results ───────────────────────────────────────────────────────
    pd.DataFrame(all_results).to_csv(out / "results.csv", index=False)
    print(f"\n  Saved: {out / 'results.csv'}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n── Generating plots ──")

    # Predicted vs true (in-dist)
    plot_pred_vs_true(
        y_id,
        {
            "A1 Ridge": pipe_a1.predict(extract(df_id, PHYSICS_NORM)),
            "A2 XGB Norm": m_a2.predict(extract(df_id, PHYSICS_NORM)),
            "A3 XGB Raw": m_a3.predict(extract(df_id, PHYSICS_RAW)),
        },
        "Experiment A — Predicted vs True eccentricity (in-distribution)",
        out / "plots" / "pred_vs_true.png",
    )

    # Generalisation curves
    plot_generalisation_curve(
        {k: v for k, v in gen_results.items() if v},
        "Experiment A — RMSE vs mass ratio q\n"
        "Key: A2 (normalised) should degrade less than A3 (raw)",
        out / "plots" / "generalisation_rmse.png",
    )

    # Residuals by eccentricity — where does the model fail?
    plot_residuals_by_e(
        y_id, m_a2.predict(extract(df_id, PHYSICS_NORM)),
        "A2 PHYSICS_NORM (in-dist)",
        out / "plots" / "residuals_by_e_a2.png",
    )
    plot_residuals_by_e(
        y_ood, m_a2.predict(extract(df_ood, PHYSICS_NORM)),
        "A2 PHYSICS_NORM (OOD)",
        out / "plots" / "residuals_by_e_a2_ood.png",
    )

    # Feature importance
    plot_feature_importance(
        m_a2, PHYSICS_NORM,
        "A2 Feature Importance — which physics features matter most?",
        out / "plots" / "feature_importance_a2.png",
    )
    plot_feature_importance(
        m_a4, ALL_PHYSICS,
        "A4 Feature Importance — PHYSICS_NORM + DYNAMICS",
        out / "plots" / "feature_importance_a4.png",
    )

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  EXPERIMENT A COMPLETE")
    print("─"*60)
    _print_table(all_results)
    print(f"\n  Output: {out}/")

    # ── Key interpretation ─────────────────────────────────────────────────
    r_a2_id  = next(r for r in all_results if "A2" in r["label"] and "in-dist" in r["label"])
    r_a2_ood = next(r for r in all_results if "A2" in r["label"] and "OOD" in r["label"])
    r_a3_ood = next(r for r in all_results if "A3" in r["label"] and "OOD" in r["label"])

    print("\n  KEY FINDING:")
    if r_a2_ood["R2"] > r_a3_ood["R2"]:
        gap = r_a2_ood["R2"] - r_a3_ood["R2"]
        print(f"  A2 (normalised) generalises better than A3 (raw).")
        print(f"  OOD R² gap = {gap:.4f}  (positive = normalisation helps)")
    else:
        print(f"  A2 and A3 perform similarly — mass normalisation effect is small.")
    print(f"  OOD degradation for A2: {r_a2_id['R2'] - r_a2_ood['R2']:.4f}  "
          f"(in-dist − OOD R²)")


def _print_table(results):
    print("  ┌──────────────────────────────────────────┬────────┬────────┬────────┐")
    print("  │ Model                                    │   R²   │  RMSE  │  MAE   │")
    print("  ├──────────────────────────────────────────┼────────┼────────┼────────┤")
    for r in results:
        lab = r["label"][:42]
        print(f"  │ {lab:<42} │ {r['R2']:6.4f} │ {r['RMSE']:6.4f} │ {r['MAE']:6.4f} │")
    print("  └──────────────────────────────────────────┴────────┴────────┴────────┘")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_CSV)
    args = p.parse_args()
    if not args.data.exists():
        print(f"ERROR: {args.data} not found. Run: python ml/dataset_generator.py")
        sys.exit(1)
    run(csv_path=args.data)